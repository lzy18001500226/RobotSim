"""Fail-closed registry and loader for the two approved X2 OmniPicker variants."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from typing import Any


REGISTRY_RELATIVE = Path("simulation/mujoco/models/agibot_x2/registry.json")
CAD_MODEL_ID = "X2-OmniPicker-CAD"
STL_MODEL_ID = "X2-OmniPicker-STL"
EXPECTED_REPOSITORY = "https://github.com/AgibotTech/agibot_x2_urdf.git"
EXPECTED_SOURCE_COMMIT = "575cc6b988f976c23550e0db85aa1e5475d3652d"
CAD_CLASSIFICATION = "SIMULATION_ONLY_UNVERIFIED_EXPERIMENTAL_GEOMETRY_SELECTED_COUPLING"
STL_CLASSIFICATION = "KINEMATIC_SIMULATION_ONLY"
CAD_EQUALITIES = {
    "official_root_mimic",
    "candidate_narrow_D",
    "candidate_wide_D",
    "candidate_full_open_narrow_q2",
    "candidate_full_open_wide_q2",
}
CAD_EXCLUSIONS = {
    frozenset(("R_hand_narrow1_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_narrow3_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_wide1_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_wide3_Link", "R_hand_wide_loop_Link")),
}
STL_EXCLUSIONS = {
    frozenset(("R_hand_narrow1_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_narrow2_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_narrow3_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_wide1_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_wide2_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_wide3_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_narrow3_Link", "R_hand_wide3_Link")),
}
STL_TABLE_JOINTS = {
    "right_claw_joint",
    "R_hand_wide1_joint",
    "R_hand_narrow3_joint",
    "R_hand_narrow_loop_joint",
    "R_hand_wide3_joint",
    "R_hand_wide_loop_joint",
}


class X2OmniPickerModelError(RuntimeError):
    """A requested model or one of its pinned assets failed validation."""


@dataclass(frozen=True)
class LoadedX2OmniPicker:
    """A validated selected model and the registry entry that governs it."""

    model_id: str
    model: Any
    classification: str
    qualification: str
    model_sha256: str
    package_path: Path
    model_path: Path
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise X2OmniPickerModelError(f"missing file required by the model contract: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_sha256(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise X2OmniPickerModelError(f"{label} SHA-256 mismatch: {path}")


def _resolve_model_file(package: Path, model_file: str, expected_sha256: str) -> Path:
    path = package / model_file
    if not path.is_file():
        raise X2OmniPickerModelError(f"missing registered model file: {path}; no fallback is permitted")
    _verify_file_sha256(path, expected_sha256, "registered model")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise X2OmniPickerModelError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2OmniPickerModelError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise X2OmniPickerModelError(f"invalid {label}: expected a JSON object")
    return value


def _root_from_checkout(repo_root: str | os.PathLike[str] | None) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]
    if not (root / ".git").exists() and not (root / ".git").is_file():
        raise X2OmniPickerModelError(f"not a RobotSim checkout: {root}")
    return root


def _inside(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise X2OmniPickerModelError(f"{label} must be repository-relative")
    result = (root / value).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise X2OmniPickerModelError(f"{label} escapes the RobotSim checkout") from exc
    return result


def _git_text(source: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise X2OmniPickerModelError("Git is required to validate the pinned vendor checkout") from exc
    if result.returncode != 0:
        raise X2OmniPickerModelError("could not validate pinned vendor checkout metadata")
    return result.stdout.strip()


def _relative_path(relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise X2OmniPickerModelError(f"{label} must be a safe relative path")
    return value


def _validate_source_checkout(root: Path, source_assets: dict[str, Any]) -> Path:
    # A developer may expose an existing, immutable reference checkout through a
    # symlink at the governed reference/ path.  The resolved target is accepted
    # only after origin, revision, clean-state, and asset hashes all verify.
    source_relative = _relative_path(source_assets["repository_path"], "vendor checkout")
    source_link = root / source_relative
    try:
        source_link.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise X2OmniPickerModelError("vendor checkout escapes the RobotSim checkout") from exc
    source = source_link.resolve()
    if not source.is_dir():
        raise X2OmniPickerModelError(
            f"missing pinned X2 source checkout: {source_link}; provision it at the path in source_assets.json"
        )
    governed_symlink = any((root.joinpath(*source_relative.parts[:depth])).is_symlink() for depth in range(1, len(source_relative.parts) + 1))
    if source != source_link and not (source_relative.parts[0] == "reference" and governed_symlink):
        raise X2OmniPickerModelError("vendor checkout resolves outside the checkout without a governed reference symlink")
    if _git_text(source, "remote", "get-url", "origin") != EXPECTED_REPOSITORY:
        raise X2OmniPickerModelError("X2 source checkout origin does not match the registered repository")
    if _git_text(source, "rev-parse", "HEAD") != EXPECTED_SOURCE_COMMIT:
        raise X2OmniPickerModelError("X2 source checkout commit does not match the pinned revision")
    if _git_text(source, "status", "--porcelain"):
        raise X2OmniPickerModelError("pinned X2 source checkout is dirty")
    for asset in [source_assets["required_assets"]["urdf"], *source_assets["required_assets"]["meshes"]]:
        path = (source / _relative_path(asset["path"], "vendor asset path")).resolve()
        if not path.is_relative_to(source):
            raise X2OmniPickerModelError("vendor asset path escapes the pinned source checkout")
        if not path.is_file():
            raise X2OmniPickerModelError(f"missing vendor asset: {asset['path']}")
        _verify_file_sha256(path, asset["sha256"], f"vendor asset {asset['path']}")
    return source


def _validate_collision_manifest(package: Path, manifest: dict[str, Any]) -> dict[str, str]:
    if manifest.get("version") != "1.0.14":
        raise X2OmniPickerModelError("collision proxy manifest does not pin CoACD 1.0.14")
    proxies: dict[str, str] = {}
    for mesh in manifest.get("mesh_decompositions", []):
        for piece in mesh.get("parts", []):
            name = piece.get("filename")
            digest = piece.get("sha256")
            if not isinstance(name, str) or Path(name).name != name or not isinstance(digest, str):
                raise X2OmniPickerModelError("invalid collision proxy manifest entry")
            if name in proxies:
                raise X2OmniPickerModelError(f"duplicate collision proxy manifest name: {name}")
            proxies[name] = digest
    if len(proxies) != 161:
        raise X2OmniPickerModelError(f"expected 161 pinned CoACD proxy assets, found {len(proxies)}")
    return proxies


def _verify_collision_assets(package: Path, directory: Path, expected: dict[str, str]) -> None:
    try:
        directory_root = directory.resolve()
        directory_root.relative_to(package.resolve())
    except ValueError as exc:
        raise X2OmniPickerModelError("collision proxy directory escapes the selected model package") from exc
    if not directory.is_dir():
        raise X2OmniPickerModelError("collision_generated is missing; run the explicit asset provisioner")
    entries = list(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise X2OmniPickerModelError("collision_generated contains a non-regular or linked asset")
    if {path.name for path in entries} != set(expected):
        raise X2OmniPickerModelError("collision_generated is incomplete or contains unregistered files")
    for name, digest in expected.items():
        _verify_file_sha256(directory / name, digest, f"collision proxy {name}")


def _load_registry(root: Path) -> dict[str, Any]:
    registry = _read_json(root / REGISTRY_RELATIVE, "X2 OmniPicker registry")
    if registry.get("schema_version") != "robotsim.x2-omnipicker-registry.v1":
        raise X2OmniPickerModelError("unsupported X2 OmniPicker registry schema")
    models = registry.get("models")
    if not isinstance(models, dict) or set(models) != {CAD_MODEL_ID, STL_MODEL_ID}:
        raise X2OmniPickerModelError("registry must contain exactly X2-OmniPicker-CAD and X2-OmniPicker-STL")
    default = registry.get("default_model_id")
    if default not in models:
        raise X2OmniPickerModelError("registry default_model_id is not a registered model")
    return registry


def selected_x2_omnipicker_model_id(*, repo_root: str | os.PathLike[str] | None = None) -> str:
    """Return the one registry-controlled default; there is no path fallback."""
    root = _root_from_checkout(repo_root)
    return str(_load_registry(root)["default_model_id"])


def _validate_cad_contract(model_path: Path, model: Any, mujoco: Any, entry: dict[str, Any]) -> None:
    if model.nu != 1 or model.nq != 8 or model.nv != 8 or model.neq != 5:
        raise X2OmniPickerModelError("compiled CAD model dimensions differ from the frozen v0.1 contract")
    if abs(float(model.opt.timestep) - 0.001) > 1e-15 or list(model.opt.gravity) != [0.0, 0.0, -9.81]:
        raise X2OmniPickerModelError("compiled CAD timestep or gravity differs from the frozen contract")
    root = ET.parse(model_path).getroot()
    equality_nodes = {node.get("name"): node for node in root.findall("./equality/*")}
    if set(equality_nodes) != CAD_EQUALITIES:
        raise X2OmniPickerModelError("CAD MJCF equality set differs from the frozen contract")
    expected_quartics = entry["quartic_equalities"]
    for branch, name in (("narrow", "candidate_full_open_narrow_q2"), ("wide", "candidate_full_open_wide_q2")):
        node = equality_nodes[name]
        spec = expected_quartics[branch]
        if node.tag != "joint" or node.get("joint1") != spec["joint1"] or node.get("joint2") != spec["joint2"]:
            raise X2OmniPickerModelError(f"{branch} CAD quartic equality joint mapping differs")
        try:
            coefficients = [float(value) for value in node.get("polycoef", "").split()]
        except ValueError as exc:
            raise X2OmniPickerModelError(f"{branch} CAD quartic equality has invalid coefficients") from exc
        if len(coefficients) != 5 or any(abs(actual - float(expected)) > 1e-12 for actual, expected in zip(coefficients, spec["coefficients_ascending"])):
            raise X2OmniPickerModelError(f"{branch} CAD quartic coefficients differ from the registry")
    exclusions = {frozenset((node.get("body1"), node.get("body2"))) for node in root.findall("./contact/exclude")}
    if exclusions != CAD_EXCLUSIONS:
        raise X2OmniPickerModelError("CAD MJCF collision exclusions differ from the frozen contract")
    if len(root.findall("./asset/mesh")) != 146:
        raise X2OmniPickerModelError("CAD MJCF visual/collision mesh reference count is incomplete")
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_right_claw_joint")
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    wide_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R_hand_wide1_joint")
    if min(actuator, root_joint, wide_joint) < 0 or int(model.actuator_trnid[actuator, 0]) != root_joint:
        raise X2OmniPickerModelError("CAD single actuator is not bound to the registered root joint")
    if (
        abs(float(model.actuator_gainprm[actuator, 0]) - 0.5) > 1e-12
        or abs(float(model.actuator_biasprm[actuator, 2]) + 0.05) > 1e-12
        or any(abs(float(actual) - expected) > 1e-12 for actual, expected in zip(model.actuator_forcerange[actuator], (-0.5, 0.5)))
        or any(abs(float(actual) - expected) > 1e-12 for actual, expected in zip(model.actuator_ctrlrange[actuator], (-1.0, 0.0)))
    ):
        raise X2OmniPickerModelError("CAD root actuator gains, force cap, or control range differ")
    _validate_root_limits(model, mujoco)


def _validate_root_limits(model: Any, mujoco: Any) -> None:
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    wide_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R_hand_wide1_joint")
    if min(root_joint, wide_joint) < 0 or not model.jnt_limited[root_joint] or not model.jnt_limited[wide_joint]:
        raise X2OmniPickerModelError("official root limits are missing")
    if any(abs(float(actual) - expected) > 1e-12 for actual, expected in zip(model.jnt_range[root_joint], (-1.0, 0.0))):
        raise X2OmniPickerModelError("narrow root limits differ from the official limits")
    if any(abs(float(actual) - expected) > 1e-12 for actual, expected in zip(model.jnt_range[wide_joint], (0.0, 1.0))):
        raise X2OmniPickerModelError("wide root limits differ from the official limits")


def _validate_stl_contract(model_path: Path, model: Any, mujoco: Any, entry: dict[str, Any], source_root: Path) -> None:
    if (model.nq, model.nv, model.nu, model.neq) != (6, 6, 0, 0):
        raise X2OmniPickerModelError("compiled STL model dimensions differ from its kinematic-table contract")
    if abs(float(model.opt.timestep) - 0.001) > 1e-15 or list(model.opt.gravity) != [0.0, 0.0, -9.81]:
        raise X2OmniPickerModelError("compiled STL timestep or gravity differs from its contract")
    root = ET.parse(model_path).getroot()
    if root.find("./actuator") is not None or root.find("./equality") is not None:
        raise X2OmniPickerModelError("STL kinematic model must not contain actuators or equality transmission guesses")
    exclusions = {frozenset((node.get("body1"), node.get("body2"))) for node in root.findall("./contact/exclude")}
    if exclusions != STL_EXCLUSIONS:
        raise X2OmniPickerModelError("STL internal collision exclusions differ from the reviewed contract")
    if {node.get("name") for node in root.findall(".//joint")} != STL_TABLE_JOINTS:
        raise X2OmniPickerModelError("STL model joints differ from the six table-driven joints")
    _validate_root_limits(model, mujoco)
    for name in ("R_hand_narrow3_joint", "R_hand_narrow_loop_joint", "R_hand_wide3_joint", "R_hand_wide_loop_joint"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not model.jnt_limited[joint_id]:
            raise X2OmniPickerModelError(f"STL table follower is missing a finite hinge range: {name}")
    for absent in ("R_hand_narrow2_joint", "R_hand_wide2_joint"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, absent) >= 0:
            raise X2OmniPickerModelError(f"STL fixed intermediate link unexpectedly exposes a joint: {absent}")
    compiler = root.find("./compiler")
    if compiler is None or compiler.get("meshdir") != "../../../../../reference/AgibotTech/agibot_x2_urdf/X2_URDF-v1.4.0/meshes":
        raise X2OmniPickerModelError("STL mesh directory does not resolve through the governed pinned source tree")
    mesh_dir = (model_path.parent / compiler.get("meshdir", "")).resolve()
    expected_mesh_dir = (source_root / "X2_URDF-v1.4.0/meshes").resolve()
    if mesh_dir != expected_mesh_dir:
        raise X2OmniPickerModelError("STL mesh directory does not resolve to the pinned source meshes")
    mesh_files = [node.get("file", "") for node in root.findall("./asset/mesh")]
    expected_meshes = {Path(asset["path"]).name for asset in _read_json(model_path.parent / entry["source_assets_manifest"], "source asset manifest")["required_assets"]["meshes"]}
    if set(mesh_files) != expected_meshes or len(mesh_files) != len(expected_meshes):
        raise X2OmniPickerModelError("STL MJCF must reference exactly the nine pinned vendor meshes")


def _validate_cad_assets(root: Path, package: Path, entry: dict[str, Any], source_assets: dict[str, Any]) -> None:
    collision_manifest_path = package / entry["collision_proxy_manifest"]
    portability_manifest_path = package / entry["portability_manifest"]
    visual_frame_manifest_path = package / entry["visual_frame_manifest"]
    _verify_file_sha256(collision_manifest_path, entry["collision_proxy_manifest_sha256"], "collision proxy manifest")
    _verify_file_sha256(portability_manifest_path, entry["portability_manifest_sha256"], "portability manifest")
    _verify_file_sha256(visual_frame_manifest_path, entry["visual_frame_manifest_sha256"], "source visual frame manifest")
    collision_manifest = _read_json(collision_manifest_path, "collision proxy manifest")
    portability = _read_json(portability_manifest_path, "portability manifest")
    _read_json(visual_frame_manifest_path, "source visual frame manifest")
    if portability.get("portable_model_sha256") != entry["portable_model_sha256"]:
        raise X2OmniPickerModelError("CAD portability manifest/model registry identity mismatch")
    if portability.get("reference_model_sha256") != entry["frozen_pr90_reference_model_sha256"]:
        raise X2OmniPickerModelError("CAD portability manifest does not bind the frozen PR #90 model")
    if collision_manifest.get("source_repo_commit") != source_assets.get("commit"):
        raise X2OmniPickerModelError("CAD collision proxies and pinned vendor source commits differ")
    if collision_manifest.get("source_urdf_sha256") != source_assets["required_assets"]["urdf"]["sha256"]:
        raise X2OmniPickerModelError("CAD collision proxy source URDF hash differs from the asset contract")
    proxy_hashes = _validate_collision_manifest(package, collision_manifest)
    _verify_collision_assets(package, package / "collision_generated", proxy_hashes)


def _validate_cad_mesh_paths(root: Path, package: Path, model_path: Path, source_root: Path) -> None:
    paths = _mesh_paths(model_path)
    for relative in paths:
        asset_path = (package / relative).resolve()
        if not asset_path.is_file():
            raise X2OmniPickerModelError(f"missing CAD model mesh asset: {relative}")
        if relative.startswith("collision_generated/"):
            if not asset_path.is_relative_to(package.resolve()):
                raise X2OmniPickerModelError("CAD collision proxy escapes its selected package")
            continue
        if relative.startswith("../../../../../reference/AgibotTech/agibot_x2_urdf/") and asset_path.is_relative_to(source_root):
            continue
        raise X2OmniPickerModelError(f"unapproved CAD model mesh path: {relative}")
    if sum(path.startswith("collision_generated/") for path in paths) != 137:
        raise X2OmniPickerModelError("CAD model does not reference the 137 required collision proxies")


def _mesh_paths(model_path: Path) -> list[str]:
    root = ET.parse(model_path).getroot()
    return [node.get("file", "") for node in root.findall("./asset/mesh")]


def load_x2_omnipicker(
    model_id: str | None = None,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> LoadedX2OmniPicker:
    """Load one exact registered model; omission selects the registry default only."""
    root = _root_from_checkout(repo_root)
    registry = _load_registry(root)
    selected_id = registry["default_model_id"] if model_id is None else model_id
    models = registry["models"]
    if selected_id not in models:
        raise X2OmniPickerModelError(f"unknown X2 OmniPicker model ID: {selected_id!r}; no fallback is permitted")
    entry = models[selected_id]
    if selected_id == CAD_MODEL_ID:
        expected_classification = CAD_CLASSIFICATION
    elif selected_id == STL_MODEL_ID:
        expected_classification = STL_CLASSIFICATION
    else:
        raise X2OmniPickerModelError("unrecognized registered X2 OmniPicker model identity")
    if entry.get("classification") != expected_classification:
        raise X2OmniPickerModelError("registered model classification differs from its selected identity")
    package = _inside(root, entry["package_path"], "package path")
    source_manifest_path = package / entry["source_assets_manifest"]
    _verify_file_sha256(source_manifest_path, entry["source_assets_manifest_sha256"], "source asset manifest")
    source_assets = _read_json(source_manifest_path, "source asset manifest")
    source_root = _validate_source_checkout(root, source_assets)
    model_path = _resolve_model_file(package, entry["model_file"], entry["portable_model_sha256"])

    if selected_id == CAD_MODEL_ID:
        _validate_cad_assets(root, package, entry, source_assets)
        _validate_cad_mesh_paths(root, package, model_path, source_root)
    else:
        trajectory_path = package / entry["trajectory_file"]
        _verify_file_sha256(trajectory_path, entry["trajectory_sha256"], "STL trajectory table")
        from .omnipicker_stl_kinematic.kinematic import KinematicTrajectory
        KinematicTrajectory.from_csv(trajectory_path)

    try:
        import mujoco
    except ImportError as exc:
        raise X2OmniPickerModelError("MuJoCo Python bindings are required; install the pinned package requirements") from exc
    if mujoco.__version__ != entry["mujoco_version"]:
        raise X2OmniPickerModelError(
            f"MuJoCo version mismatch: expected {entry['mujoco_version']}, got {mujoco.__version__}"
        )
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as exc:
        raise X2OmniPickerModelError("MuJoCo failed to compile the registered model") from exc
    if selected_id == CAD_MODEL_ID:
        _validate_cad_contract(model_path, model, mujoco, entry)
    else:
        _validate_stl_contract(model_path, model, mujoco, entry, source_root)
    return LoadedX2OmniPicker(
        model_id=selected_id,
        model=model,
        classification=entry["classification"],
        qualification=entry["qualification"],
        model_sha256=entry["portable_model_sha256"],
        package_path=package,
        model_path=model_path,
        metadata=entry,
    )


def load_default_x2_omnipicker(*, repo_root: str | os.PathLike[str] | None = None) -> LoadedX2OmniPicker:
    """Load precisely the default model recorded in the registry."""
    return load_x2_omnipicker(None, repo_root=repo_root)
