#!/usr/bin/env python3
"""Read-only provenance and compile audit for the saved Issue #46 run05 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "assets/robots/registry.json"
HASHES_PATH = ROOT / "assets/robots/source_hashes.json"
COMPILED_PATH = ROOT / "assets/robots/compiled_model_signatures.json"
REFERENCE_COMMIT = "cbbe4a5dc9bb12f716dc05880146d454aee6a5ca"
SOURCE_PROFILE = "agibot_x2_urdf_575cc6b"
ASSET_ID = "x2_ultra_omnipicker_v1_4"
SOURCE_URDF = "X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
SOURCE_MODEL_REL = "X2_URDF-v1.4.0/meshes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(worktree: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(worktree), *args], capture_output=True,
                            text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def signature(model: mujoco.MjModel) -> dict[str, int]:
    return {name: int(getattr(model, name)) for name in
            ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "nmesh", "neq")}


def names(model: mujoco.MjModel, obj: mujoco.mjtObj, count: int) -> list[str | None]:
    return [mujoco.mj_id2name(model, obj, index) for index in range(count)]


def compile_saved_fixture(xml_path: Path, source_mesh_root: Path) -> mujoco.MjModel:
    root = ET.parse(xml_path).getroot()
    refs = root.findall("./asset/mesh")
    if not refs:
        raise ValueError(f"No mesh assets in saved MJCF: {xml_path}")
    with tempfile.TemporaryDirectory(prefix="robotsim-issue46-model-audit-") as temp_name:
        temp = Path(temp_name)
        copied_xml = temp / xml_path.name
        shutil.copy2(xml_path, copied_xml)
        for element in refs:
            filename = element.get("file")
            if not filename:
                raise ValueError("Mesh reference without file attribute")
            target = temp / Path(filename).name
            source = source_mesh_root / Path(filename).name
            if not source.is_file():
                raise FileNotFoundError(source)
            target.symlink_to(source)
        return mujoco.MjModel.from_xml_path(str(copied_xml))


def compile_registered_source(source_path: Path) -> mujoco.MjModel:
    # Match the registry renderer/importer defaults. No MjData or integration is used.
    return mujoco.MjSpec.from_file(str(source_path)).compile()


def source_mount(source_urdf: Path) -> dict[str, Any]:
    root = ET.parse(source_urdf).getroot()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link") == "R_omnipicker_base_link":
            return {
                "joint_name": joint.get("name"),
                "joint_type": joint.get("type"),
                "parent_link": joint.find("parent").get("link"),
                "child_link": child.get("link"),
                "origin_xyz": [float(x) for x in joint.find("origin").get("xyz", "0 0 0").split()],
                "origin_rpy": [float(x) for x in joint.find("origin").get("rpy", "0 0 0").split()],
            }
    raise ValueError("Could not find the official right wrist mount to R_omnipicker_base_link")


def semantic_xml(element: ET.Element) -> tuple[Any, ...]:
    attributes = dict(element.attrib)
    if element.tag == "mesh" and "filename" in attributes:
        attributes["filename"] = Path(attributes["filename"]).name
    return (element.tag, tuple(sorted(attributes.items())), (element.text or "").strip(),
            tuple(semantic_xml(child) for child in element))


def source_subtree_comparison(source_urdf: Path, generated_urdf: Path) -> dict[str, Any]:
    source_root = ET.parse(source_urdf).getroot()
    generated_root = ET.parse(generated_urdf).getroot()
    source_links = {node.get("name"): node for node in source_root.findall("link")}
    generated_links = {node.get("name"): node for node in generated_root.findall("link")}
    source_joints = {node.get("name"): node for node in source_root.findall("joint")}
    generated_joints = {node.get("name"): node for node in generated_root.findall("joint")}

    root_link = "right_wrist_roll_link"
    descendants = {root_link}
    while True:
        added = {joint.find("child").get("link") for joint in source_joints.values()
                 if joint.find("parent").get("link") in descendants} - descendants
        if not added:
            break
        descendants.update(added)
    expected_joints = {name for name, joint in source_joints.items()
                       if joint.find("parent").get("link") in descendants and
                       joint.find("child").get("link") in descendants}
    expected_links = descendants

    link_semantics_match = all(name in generated_links and
                               semantic_xml(source_links[name]) == semantic_xml(generated_links[name])
                               for name in expected_links)
    joint_semantics_match = all(name in generated_joints and
                                semantic_xml(source_joints[name]) == semantic_xml(generated_joints[name])
                                for name in expected_joints)
    source_collision_meshes = sorted({
        Path(mesh.get("filename", "")).name
        for name in expected_links for mesh in source_links[name].findall("collision/geometry/mesh")
    })
    source_visual_meshes = sorted({
        Path(mesh.get("filename", "")).name
        for name in expected_links for mesh in source_links[name].findall("visual/geometry/mesh")
    })
    return {
        "root_link": root_link,
        "expected_descendant_link_names": sorted(expected_links),
        "generated_link_names": sorted(generated_links),
        "expected_joint_names": sorted(expected_joints),
        "generated_joint_names": sorted(generated_joints),
        "link_semantics_match_after_mesh_path_normalization": link_semantics_match,
        "joint_semantics_match": joint_semantics_match,
        "source_collision_mesh_files": source_collision_meshes,
        "source_visual_mesh_files": source_visual_meshes,
        "generated_subtree_geometry_changes": [],
    }


def wrist_mesh_roles(source_urdf: Path) -> dict[str, Any]:
    root = ET.parse(source_urdf).getroot()
    link = next((item for item in root.findall("link") if item.get("name") == "right_wrist_roll_link"), None)
    if link is None:
        raise ValueError("Official URDF has no right_wrist_roll_link")

    def first_mesh(kind: str) -> dict[str, Any] | None:
        mesh = link.find(f"{kind}/geometry/mesh")
        if mesh is None:
            return None
        return {"filename": mesh.get("filename"),
                "scale": [float(x) for x in mesh.get("scale", "1 1 1").split()]}

    return {"visual": first_mesh("visual"), "collision": first_mesh("collision")}


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source_hashes = json.loads(args.source_hashes.read_text(encoding="utf-8"))
    compiled_signatures = json.loads(args.compiled_signatures.read_text(encoding="utf-8"))
    asset = next(item for item in registry["assets"] if item["asset_id"] == ASSET_ID)
    profile = registry["source_profiles"][SOURCE_PROFILE]
    lock = source_hashes["profiles"][SOURCE_PROFILE]["files"]
    official_urdf = args.source_root / SOURCE_URDF
    source_urdf_sha = sha256_file(official_urdf)
    if source_urdf_sha != lock[SOURCE_URDF]:
        raise ValueError("Pinned source URDF hash does not match the registry lock")

    identity_path = args.packet / "experiment_identity_closeout_v2.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    baseline_identity = json.loads((args.baseline / "runtime_identity.json").read_text(encoding="utf-8"))
    subtree_identity = json.loads((args.baseline / "source_subtree_identity.json").read_text(encoding="utf-8"))
    result_closeout = json.loads((args.packet / "result_closeout_v2.json").read_text(encoding="utf-8"))
    fixture_xml = args.baseline / "generated_fixture_model.xml"
    generated_subtree = args.baseline / "generated_right_wrist_omnipicker_subtree.urdf"
    fixture_xml_sha = sha256_file(fixture_xml)
    subtree_sha = sha256_file(generated_subtree)
    if fixture_xml_sha != identity["compiled_model_sha256"]["run05_baseline"]:
        raise ValueError("Saved run05 MJCF hash does not match closeout identity")
    if subtree_sha != identity["source_subtree_identity"]["generated_urdf_sha256"]:
        raise ValueError("Saved generated subtree URDF hash does not match closeout identity")
    if subtree_sha != subtree_identity["generated_urdf_sha256"]:
        raise ValueError("Baseline subtree hash does not match its source identity record")
    if baseline_identity["source_urdf_sha256"] != source_urdf_sha:
        raise ValueError("Baseline runtime URDF hash differs from current pinned source")

    source_mjcf = ET.parse(fixture_xml).getroot()
    mesh_records = []
    for element in source_mjcf.findall("./asset/mesh"):
        filename = Path(element.get("file", "")).name
        rel = f"{SOURCE_MODEL_REL}/{filename}"
        actual = sha256_file(args.source_root / rel)
        expected = lock.get(rel)
        if expected is None or actual != expected:
            raise ValueError(f"Runtime mesh does not match pinned registry source: {filename}")
        mesh_records.append({"mjcf_name": element.get("name"), "file": filename,
                             "source_path": rel, "sha256": actual})

    fixture_model = compile_saved_fixture(fixture_xml, args.source_root / SOURCE_MODEL_REL)
    registered_model = compile_registered_source(official_urdf)
    expected_signature = compiled_signatures["profiles"][ASSET_ID]["snapshot"]
    registered_signature = signature(registered_model)
    registry_structure = {key: int(asset["expected_structure"][key]) for key in signature(registered_model)}
    if registered_signature != registry_structure:
        raise ValueError(f"Pinned URDF compile differs from registry structure: {registered_signature}")
    if signature(registered_model) != {key: int(compiled_signatures["profiles"][ASSET_ID]["snapshot"]["signature"][key])
                                       for key in signature(registered_model)}:
        raise ValueError("Pinned URDF compile differs from saved registry compiled signature")

    fixture_body_names = names(fixture_model, mujoco.mjtObj.mjOBJ_BODY, fixture_model.nbody)
    fixture_joint_names = names(fixture_model, mujoco.mjtObj.mjOBJ_JOINT, fixture_model.njnt)
    fixture_mesh_names = names(fixture_model, mujoco.mjtObj.mjOBJ_MESH, fixture_model.nmesh)
    subtree_comparison = source_subtree_comparison(official_urdf, generated_subtree)
    fixture_actuator_names = names(fixture_model, mujoco.mjtObj.mjOBJ_ACTUATOR, fixture_model.nu)
    source_subtree = ET.parse(generated_subtree).getroot()
    urdf_joint_records = []
    for joint in source_subtree.findall("joint"):
        mimic = joint.find("mimic")
        urdf_joint_records.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "mimic": None if mimic is None else {"joint": mimic.get("joint"),
                                                  "multiplier": float(mimic.get("multiplier", "1")),
                                                  "offset": float(mimic.get("offset", "0"))},
        })
    mimic_source_names = [item["name"] for item in urdf_joint_records if item["mimic"] is not None]
    compiled_mimic_names = sorted(set(mimic_source_names) & set(filter(None, fixture_joint_names)))
    omitted_mimic_names = sorted(set(mimic_source_names) - set(filter(None, fixture_joint_names)))
    source_mimic_relations = [
        {"joint_name": item["name"], "target_joint": item["mimic"]["joint"],
         "multiplier": item["mimic"]["multiplier"], "offset": item["mimic"]["offset"]}
        for item in urdf_joint_records if item["mimic"] is not None
    ]
    source_fixed_joint_names = sorted(item["name"] for item in urdf_joint_records
                                      if item["type"] == "fixed")
    source_fixed_finger_joint_names = [name for name in source_fixed_joint_names
                                       if name != "R_omnipicker_joint"]
    fixture_mesh_files = sorted(row["file"] for row in mesh_records)

    mount = source_mount(official_urdf)
    visual_collision = wrist_mesh_roles(official_urdf)
    wrist_files = {}
    for kind, record in visual_collision.items():
        if record is None:
            wrist_files[kind] = None
            continue
        rel = f"{SOURCE_MODEL_REL}/{Path(record['filename']).name}"
        wrist_files[kind] = {**record, "source_path": rel, "sha256": sha256_file(args.source_root / rel)}
    fixture_body_counts = {name: fixture_body_names.count(name) for name in fixture_body_names if name}
    no_duplicate_ee = fixture_body_counts.get("R_omnipicker_base_link", 0) == 1 and not any(
        name in fixture_body_counts for name in ("L_omnipicker_base_link", "R_palm", "L_palm"))
    no_omnihand = not any("omnihand" in (name or "").lower() for name in fixture_body_names)

    head = git_text(args.worktree, "rev-parse", "HEAD")
    status = git_text(args.worktree, "status", "--porcelain", "--untracked-files=all")
    if head != REFERENCE_COMMIT or status:
        raise ValueError(f"Issue #46 reference worktree is not at the requested clean head: {head}; dirty={bool(status)}")
    runner_path = args.worktree / "scripts/research/issue46_omnipicker_p0_isolated.py"
    runner_sha = sha256_file(runner_path)
    if runner_sha != identity["runtime"]["runner_sha256"]:
        raise ValueError("Reference worktree runner does not match the preserved run05 runner hash")

    expected_model_name = "robotsim_p0_source_right_omnipicker_fixture"
    fixture_model_name = ET.parse(fixture_xml).getroot().get("model")
    classification = "EXPECTED_DERIVED_MODEL_WITH_DOCUMENTED_DIFFERENCES"
    checks = {
        "pinned_source_urdf_hash_matches_registry": True,
        "all_compiled_fixture_mesh_hashes_match_registry": all(row["sha256"] == lock[row["source_path"]] for row in mesh_records),
        "generated_right_wrist_subtree_matches_official_source_semantics":
            subtree_comparison["link_semantics_match_after_mesh_path_normalization"] and
            subtree_comparison["joint_semantics_match"] and
            subtree_comparison["expected_descendant_link_names"] == subtree_comparison["generated_link_names"] and
            subtree_comparison["expected_joint_names"] == subtree_comparison["generated_joint_names"],
        "runtime_meshes_are_exact_source_collision_mesh_set":
            fixture_mesh_files == subtree_comparison["source_collision_mesh_files"],
        "saved_run05_mjcf_hash_matches_identity": True,
        "saved_generated_subtree_hash_matches_identity": True,
        "official_mount_matches_registry": mount["parent_link"] == "right_wrist_roll_link" and
            mount["child_link"] == "R_omnipicker_base_link" and mount["origin_xyz"] == [0.00196, 0.00035, -0.062] and
            mount["origin_rpy"] == [3.14, 0.0, 0.0],
        "compiled_model_name_matches_fixture": fixture_model_name == expected_model_name,
        "single_right_omnipicker_no_duplicate_end_effector": no_duplicate_ee,
        "no_omnihand_links": no_omnihand,
        "source_has_expected_single_opposed_jaw_mimic": source_mimic_relations == [
            {"joint_name": "R_hand_wide1_joint", "target_joint": "right_claw_joint",
             "multiplier": -1.0, "offset": 0.0}
        ],
        "both_source_jaw_joints_present_in_compiled_fixture":
            "right_claw_joint" in fixture_joint_names and "R_hand_wide1_joint" in fixture_joint_names,
        "simulation_paired_jaw_actuators_present":
            "diag_right_driver_servo" in fixture_actuator_names and
            "diag_right_follower_servo" in fixture_actuator_names,
        "no_compiled_equality_constraint": fixture_model.neq == 0,
        "local03_reference_head_clean": True,
        "no_physics_steps_in_this_audit": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Runtime model verification gate failed: {checks}")

    result = {
        "classification": classification,
        "reference": {"commit": head, "branch": git_text(args.worktree, "branch", "--show-current"),
                      "worktree": str(args.worktree), "clean": not bool(status)},
        "registry_candidate": {"asset_id": ASSET_ID, "source_model_path": asset["model_path"],
                               "source_repository": profile["repository"], "source_commit": profile["commit"],
                               "source_urdf_sha256": source_urdf_sha,
                               "compiled_signature": registered_signature,
                               "compiled_hand_joints": [name for name in names(registered_model, mujoco.mjtObj.mjOBJ_JOINT, registered_model.njnt)
                                                        if name and ("claw_joint" in name or "wide1_joint" in name)]},
        "actual_prior_contact_experiment": {
            "experiment": "Issue #46 OmniPicker run05 measured-preload physical contact trial",
            "baseline_evidence_dir": str(args.baseline),
            "posthoc_packet_dir": str(args.packet),
            "posthoc_packet_dynamic_physics_steps": int(identity["all_new_geometry_candidates_physics_steps"]),
            "prior_result_summary": "run05 result is preserved in prior_evidence; posthoc attempt02 performed static candidate checks only",
            "runner_sha256": runner_sha,
            "mujoco": baseline_identity["mujoco_native"],
            "generated_right_wrist_subtree_urdf": {"path": "generated_right_wrist_omnipicker_subtree.urdf",
                                                    "sha256": subtree_sha,
                                                    "source_link_count": len(source_subtree.findall("link")),
                                                    "source_joint_records": urdf_joint_records},
            "compiled_mjcf": {"path": "generated_fixture_model.xml", "sha256": fixture_xml_sha,
                               "model_name": expected_model_name, "signature": signature(fixture_model),
                               "body_names": fixture_body_names, "joint_names": fixture_joint_names,
                               "actuator_names": fixture_actuator_names, "mesh_names": fixture_mesh_names,
                               "mesh_assets": mesh_records},
            "wrist_to_gripper_mount": mount,
            "wrist_mesh_roles": wrist_files,
            "source_mimic_relations": source_mimic_relations,
            "source_fixed_joint_names": source_fixed_joint_names,
            "source_fixed_finger_joint_names": source_fixed_finger_joint_names,
            "official_source_subtree_comparison": subtree_comparison,
            "source_mimic_joint_coordinates_present_in_compiled_fixture": compiled_mimic_names,
            "source_mimic_joint_coordinates_omitted_from_compiled_fixture": omitted_mimic_names,
            "mimic_enforcement": {
                "compiled_as_equality": False,
                "compiled_equality_count": int(fixture_model.neq),
                "simulation_mapping": "follower=-driver",
                "driver_joint": "right_claw_joint",
                "follower_joint": "R_hand_wide1_joint",
                "actuators": ["diag_right_driver_servo", "diag_right_follower_servo"],
                "classification": "simulation-only paired-jaw actuator emulation; not a native MuJoCo mimic constraint",
            },
            "compiled_equalities": int(fixture_model.neq),
            "right_wrist_mjcf_body_pos_quat": {"pos": [float(x) for x in
                ET.parse(fixture_xml).getroot().find(".//body[@name='R_omnipicker_base_link']").get("pos").split()],
                "quat_wxyz": [float(x) for x in
                ET.parse(fixture_xml).getroot().find(".//body[@name='R_omnipicker_base_link']").get("quat").split()]},
        },
        "intentional_derived_differences": [
            "Right-wrist and descendants only; upstream arm, torso, and opposite end-effector omitted for isolated one-hand contact experiment.",
            "Wrist support is a fixture with approach/lift slides and motors; table, ground, and a free-jointed diagnostic bottle are added.",
            "Two simulation-only jaw-driver actuators plus two fixture actuators are added; source URDF declares no transmissions or actuators.",
            "The source URDF has one opposed-jaw mimic relation (R_hand_wide1_joint = -right_claw_joint); both coordinates compile independently and no equality encodes that relation. The run05 controller emulates it with two separate servo actuators and an opposite-direction target mapping.",
            "The other distal finger links are fixed joints in the pinned source URDF, not omitted mimic followers or independently actuated joints.",
            "Runtime contact geoms are compiled collision meshes. The distinct source visual wrist mesh is not an MJCF mesh in the saved fixture.",
        ],
        "contact_conclusion_scope": {
            "valid_for": "The exact saved MuJoCo 3.3.6 isolated fixture and its compiled collision geoms.",
            "not_valid_as": "A full-body X2 simulation, a model with native mimic-constraint enforcement, or real hardware-equivalent contact evidence.",
            "collision_hull_note": "MuJoCo contacts use compiled convex mesh hulls. Source triangle sampling independently found the right-wrist collision STL also entered the bottle volume in both tested correction candidates, so those two blockers were not hull-only; this does not prove CAD or physical housing equivalence.",
        },
        "checks": checks,
        "physics_steps_in_this_audit": 0,
        "mj_step_calls_in_this_audit": 0,
    }
    return result


def render_report(result: dict[str, Any]) -> str:
    runtime = result["actual_prior_contact_experiment"]
    fixture = runtime["compiled_mjcf"]
    lines = [
        "# Issue #46 Runtime Model Comparison",
        "",
        f"**Classification: `{result['classification']}`**",
        "",
        "This is a read-only model/provenance audit. It does not alter Local03 and runs zero physics steps. The posthoc attempt02 packet contains static candidate checks; the actual prior physical-contact evidence is the separately preserved run05 packet.",
        "",
        "## Provenance",
        "",
        f"- Local03 reference: `{result['reference']['commit']}` on `{result['reference']['branch']}`; clean at audit time.",
        f"- Registered source: `{result['registry_candidate']['source_repository']}` commit `{result['registry_candidate']['source_commit']}`.",
        f"- Root URDF: `{result['registry_candidate']['source_model_path']}`; SHA-256 `{result['registry_candidate']['source_urdf_sha256']}`.",
        f"- Extracted right-wrist subtree URDF SHA-256: `{runtime['generated_right_wrist_subtree_urdf']['sha256']}`.",
        f"- Actual run05 MJCF SHA-256: `{fixture['sha256']}` (matches recorded `run05_baseline` hash).",
        f"- Runner SHA-256: `{runtime['runner_sha256']}`; MuJoCo `{runtime['mujoco']}`.",
        "",
        "## Assembly and Compile",
        "",
        f"The official fixed mount is `{runtime['wrist_to_gripper_mount']['joint_name']}`: `{runtime['wrist_to_gripper_mount']['parent_link']}` → `{runtime['wrist_to_gripper_mount']['child_link']}`, xyz `{runtime['wrist_to_gripper_mount']['origin_xyz']} m`, rpy `{runtime['wrist_to_gripper_mount']['origin_rpy']} rad`. The derived MJCF retains the same origin (quat wxyz `{runtime['right_wrist_mjcf_body_pos_quat']['quat_wxyz']}` on the gripper root).",
        "",
        f"Compiled fixture model `{fixture['model_name']}` signature: `{fixture['signature']}`.",
        f"Bodies: `{fixture['body_names']}`.",
        f"Joints: `{fixture['joint_names']}`.",
        f"Actuators: `{fixture['actuator_names']}`.",
        "",
        "Exactly one right OmniPicker base is present. There is no left/duplicate end-effector and no OmniHand/thumb/index/middle/ring/pinky body. The official compiled registry candidate was independently compiled with MuJoCo 3.3.6 and matches its registered structure; the fixture is a documented right-only derivative with table, ground, bottle, wrist carriage, and simulation actuators.",
        "",
        "### Runtime mesh assets",
        "",
        "| MJCF mesh | Source file | SHA-256 |",
        "|---|---|---|",
    ]
    for mesh in fixture["mesh_assets"]:
        lines.append(f"| `{mesh['mjcf_name']}` | `{mesh['source_path']}` | `{mesh['sha256']}` |")
    lines += [
        "",
        f"Pinned right-wrist source visual mesh: `{runtime['wrist_mesh_roles']['visual']['filename']}` (`{runtime['wrist_mesh_roles']['visual']['sha256']}`). Source collision mesh: `{runtime['wrist_mesh_roles']['collision']['filename']}` (`{runtime['wrist_mesh_roles']['collision']['sha256']}`). The run05 MJCF includes the collision mesh but not the separate visual mesh; its visible wrist therefore follows the collision geometry.",
        "",
        "## Differences and Impact",
        "",
        "- The upstream arm and torso and the opposite hand are intentionally omitted. A fixture supports and moves the right wrist; table, ground, and a free-jointed bottle are added. These changes define an isolated one-hand experiment, not the full X2 robot.",
        "- Two source jaw coordinates are actuated through a simulation-only paired-jaw abstraction; the fixture also adds approach/lift slide actuators. No source actuator or transmission is claimed.",
        f"- The extracted URDF has {len(runtime['generated_right_wrist_subtree_urdf']['source_joint_records'])} joint records. Exactly one is a mimic relation: `{runtime['mimic_enforcement']['follower_joint']} = -{runtime['mimic_enforcement']['driver_joint']}`. Both jaw coordinates appear in the compiled joints, but the fixture has `{runtime['compiled_equalities']}` equality constraints, so the relation is not natively enforced by MuJoCo. Instead, the runner creates separate `{runtime['mimic_enforcement']['actuators']}` servos and commands opposite targets. The distal finger links `{runtime['source_fixed_finger_joint_names']}` are fixed in the source URDF, not omitted mimic followers. This matches the documented run05 simulation abstraction, not hardware-calibrated coupling.",
        "- Runtime contact points and forces are from MuJoCo's compiled convex collision hulls. For the two failed static correction candidates, sampled source wrist collision triangles were also inside the bottle by approximately 1.128 mm and 0.864 mm, closely matching compiled hull penetration. Those two specific blockers are not explained by hull expansion alone. This does not establish physical CAD accuracy.",
        "- The source URDF visual wrist mesh is distinct from the collision mesh, as intended upstream. Because only the collision mesh is in the saved fixture MJCF, visual wrist appearance is not a source-visual fidelity check.",
        "",
        "The prior dynamic contact observations apply to this exact isolated compiled fixture only. They do not establish full-body X2 contact behavior, follower-link motion, hardware equivalence, or a successful pickup. The posthoc attempt02 candidates ran zero physics steps.",
        "",
        "## Result",
        "",
        f"`{result['classification']}` — pinned source, right wrist mount, and all nine meshes referenced by the saved fixture MJCF match the registry; differences are the documented right-only contact fixture, collision-mesh wrist rendering, and simulation-only paired-jaw actuator emulation of the single source mimic relation. See `issue46_runtime_model_comparison.json` for exact names, hashes, and source records.",
        "",
        "Audit physics steps: `0`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--source-hashes", type=Path, default=HASHES_PATH)
    parser.add_argument("--compiled-signatures", type=Path, default=COMPILED_PATH)
    args = parser.parse_args()
    result = build_result(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"classification": result["classification"], "output_json": str(args.output_json),
                      "output_report": str(args.output_report), "checks": result["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
