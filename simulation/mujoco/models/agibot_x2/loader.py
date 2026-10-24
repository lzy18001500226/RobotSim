"""Fail-closed loader for explicitly registered X2 OmniPicker simulation models."""
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
EXPECTED_REPOSITORY = "https://github.com/AgibotTech/agibot_x2_urdf.git"
EXPECTED_SOURCE_COMMIT = "575cc6b988f976c23550e0db85aa1e5475d3652d"
EXPECTED_EQUALITIES = {
    "official_root_mimic",
    "candidate_narrow_D",
    "candidate_wide_D",
    "candidate_full_open_narrow_q2",
    "candidate_full_open_wide_q2",
}
EXPECTED_EXCLUSIONS = {
    frozenset(("R_hand_narrow1_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_narrow3_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_wide1_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_wide3_Link", "R_hand_wide_loop_Link")),
}


class X2OmniPickerModelError(RuntimeError):
    """A requested model or one of its pinned assets failed validation."""


@dataclass(frozen=True)
class LoadedX2OmniPicker:
    model_id: str
    model: Any
    classification: str
    qualification: str
    model_sha256: str


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
    _verify_file_sha256(path, expected_sha256, "portable model")
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


def _validate_source_checkout(root: Path, source_assets: dict[str, Any]) -> Path:
    source = _inside(root, source_assets["repository_path"], "vendor checkout")
    if not source.is_dir():
        raise X2OmniPickerModelError(
            f"missing pinned X2 source checkout: {source}; provision it at the path in source_assets.json"
        )
    if _git_text(source, "remote", "get-url", "origin") != EXPECTED_REPOSITORY:
        raise X2OmniPickerModelError("X2 source checkout origin does not match the registered repository")
    if _git_text(source, "rev-parse", "HEAD") != EXPECTED_SOURCE_COMMIT:
        raise X2OmniPickerModelError("X2 source checkout commit does not match the pinned revision")
    if _git_text(source, "status", "--porcelain"):
        raise X2OmniPickerModelError("pinned X2 source checkout is dirty")
    for asset in [source_assets["required_assets"]["urdf"], *source_assets["required_assets"]["meshes"]]:
        path = _inside(source, asset["path"], "vendor asset path")
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


def _validate_model_contract(model_path: Path, model: Any, mujoco: Any, entry: dict[str, Any]) -> None:
    if model.nu != 1 or model.nq != 8 or model.nv != 8 or model.neq != 5:
        raise X2OmniPickerModelError("compiled model dimensions differ from the registered v0.1 contract")
    if abs(float(model.opt.timestep) - 0.001) > 1e-15 or list(model.opt.gravity) != [0.0, 0.0, -9.81]:
        raise X2OmniPickerModelError("compiled timestep or gravity differs from the registered v0.1 contract")

    root = ET.parse(model_path).getroot()
    equality_nodes = {node.get("name"): node for node in root.findall("./equality/*")}
    equality_names = set(equality_nodes)
    if equality_names != EXPECTED_EQUALITIES:
        raise X2OmniPickerModelError("MJCF equality set differs from the registered v0.1 contract")
    expected_quartics = entry["quartic_equalities"]
    for branch, name in (("narrow", "candidate_full_open_narrow_q2"), ("wide", "candidate_full_open_wide_q2")):
        node = equality_nodes[name]
        spec = expected_quartics[branch]
        if node.tag != "joint" or node.get("joint1") != spec["joint1"] or node.get("joint2") != spec["joint2"]:
            raise X2OmniPickerModelError(f"{branch} quartic equality joint mapping differs")
        try:
            coefficients = [float(x) for x in node.get("polycoef", "").split()]
        except ValueError as exc:
            raise X2OmniPickerModelError(f"{branch} quartic equality has invalid coefficients") from exc
        if len(coefficients) != 5 or any(abs(a - float(b)) > 1e-12 for a, b in zip(coefficients, spec["coefficients_ascending"])):
            raise X2OmniPickerModelError(f"{branch} quartic coefficients differ from the registry")
    exclusions = {
        frozenset((node.get("body1"), node.get("body2")))
        for node in root.findall("./contact/exclude")
    }
    if exclusions != EXPECTED_EXCLUSIONS:
        raise X2OmniPickerModelError("MJCF collision exclusions differ from the registered v0.1 contract")
    if len(root.findall("./asset/mesh")) != 146:
        raise X2OmniPickerModelError("MJCF visual/collision mesh reference count is incomplete")

    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_right_claw_joint")
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
    wide_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R_hand_wide1_joint")
    if min(actuator, root_joint, wide_joint) < 0 or int(model.actuator_trnid[actuator, 0]) != root_joint:
        raise X2OmniPickerModelError("single actuator is not bound to the registered root joint")
    if (
        abs(float(model.actuator_gainprm[actuator, 0]) - 0.5) > 1e-12
        or abs(float(model.actuator_biasprm[actuator, 2]) + 0.05) > 1e-12
        or any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(model.actuator_forcerange[actuator], (-0.5, 0.5)))
        or any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(model.actuator_ctrlrange[actuator], (-1.0, 0.0)))
    ):
        raise X2OmniPickerModelError("root actuator gains, force cap, or control range differ")
    if not model.jnt_limited[root_joint] or not model.jnt_limited[wide_joint]:
        raise X2OmniPickerModelError("official root limits are missing")
    if any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(model.jnt_range[root_joint], (-1.0, 0.0))):
        raise X2OmniPickerModelError("narrow root limits differ from the official limits")
    if any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(model.jnt_range[wide_joint], (0.0, 1.0))):
        raise X2OmniPickerModelError("wide root limits differ from the official limits")


def load_x2_omnipicker(
    model_id: str,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> LoadedX2OmniPicker:
    """Load one exact registered model; unknown IDs and missing assets fail closed."""
    root = _root_from_checkout(repo_root)
    registry = _read_json(root / REGISTRY_RELATIVE, "X2 OmniPicker registry")
    if registry.get("schema_version") != "robotsim.x2-omnipicker-registry.v1":
        raise X2OmniPickerModelError("unsupported X2 OmniPicker registry schema")
    models = registry.get("models")
    if not isinstance(models, dict) or model_id not in models:
        raise X2OmniPickerModelError(f"unknown X2 OmniPicker model ID: {model_id!r}; no fallback is permitted")
    entry = models[model_id]
    if entry.get("classification") != "SIMULATION_ONLY_UNVERIFIED_EXPERIMENTAL_GEOMETRY_SELECTED_COUPLING":
        raise X2OmniPickerModelError("registered model classification is not the expected simulation-only scope")

    package = _inside(root, entry["package_path"], "package path")
    source_manifest_path = package / entry["source_assets_manifest"]
    collision_manifest_path = package / entry["collision_proxy_manifest"]
    portability_manifest_path = package / entry["portability_manifest"]
    visual_frame_manifest_path = package / entry["visual_frame_manifest"]
    _verify_file_sha256(source_manifest_path, entry["source_assets_manifest_sha256"], "source asset manifest")
    _verify_file_sha256(collision_manifest_path, entry["collision_proxy_manifest_sha256"], "collision proxy manifest")
    _verify_file_sha256(portability_manifest_path, entry["portability_manifest_sha256"], "portability manifest")
    _verify_file_sha256(visual_frame_manifest_path, entry["visual_frame_manifest_sha256"], "source visual frame manifest")
    source_assets = _read_json(source_manifest_path, "source asset manifest")
    collision_manifest = _read_json(collision_manifest_path, "collision proxy manifest")
    portability = _read_json(portability_manifest_path, "portability manifest")
    _read_json(visual_frame_manifest_path, "source visual frame manifest")
    if portability.get("portable_model_sha256") != entry["portable_model_sha256"]:
        raise X2OmniPickerModelError("portability manifest/model registry identity mismatch")
    if portability.get("reference_model_sha256") != entry["frozen_pr90_reference_model_sha256"]:
        raise X2OmniPickerModelError("portability manifest does not bind the frozen PR #90 model")
    if collision_manifest.get("source_repo_commit") != source_assets.get("commit"):
        raise X2OmniPickerModelError("collision proxies and pinned vendor source commits differ")
    if collision_manifest.get("source_urdf_sha256") != source_assets["required_assets"]["urdf"]["sha256"]:
        raise X2OmniPickerModelError("collision proxy source URDF hash differs from the asset contract")
    source_root = _validate_source_checkout(root, source_assets)

    model_path = _resolve_model_file(package, entry["model_file"], entry["portable_model_sha256"])

    proxy_hashes = _validate_collision_manifest(package, collision_manifest)
    collision_directory = package / "collision_generated"
    _verify_collision_assets(package, collision_directory, proxy_hashes)
    for relative in _mesh_paths(model_path):
        asset_path = (package / relative).resolve()
        try:
            asset_path.relative_to(root.resolve())
        except ValueError as exc:
            raise X2OmniPickerModelError("model mesh path escapes the provisioned repository roots") from exc
        if not asset_path.is_file():
            raise X2OmniPickerModelError(f"missing model mesh asset: {relative}")
        if relative.startswith("collision_generated/"):
            expected = proxy_hashes.get(Path(relative).name)
            if expected is None or _sha256(asset_path) != expected:
                raise X2OmniPickerModelError(f"collision proxy missing from manifest or SHA mismatch: {relative}")
        elif relative.startswith("../../../../../reference/AgibotTech/agibot_x2_urdf/"):
            if not asset_path.is_relative_to(source_root):
                raise X2OmniPickerModelError("vendor mesh path does not resolve under the pinned source checkout")
        else:
            raise X2OmniPickerModelError(f"unapproved model mesh path: {relative}")

    referenced_proxy_count = sum(path.startswith("collision_generated/") for path in _mesh_paths(model_path))
    if referenced_proxy_count != 137:
        raise X2OmniPickerModelError(f"expected 137 collision proxies, found {referenced_proxy_count}")

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
    _validate_model_contract(model_path, model, mujoco, entry)
    return LoadedX2OmniPicker(
        model_id=model_id,
        model=model,
        classification=entry["classification"],
        qualification=entry["qualification"],
        model_sha256=entry["portable_model_sha256"],
    )


def _mesh_paths(model_path: Path) -> list[str]:
    root = ET.parse(model_path).getroot()
    return [node.get("file", "") for node in root.findall("./asset/mesh")]
