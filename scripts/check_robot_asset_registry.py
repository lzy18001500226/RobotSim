#!/usr/bin/env python3
"""Offline identity checker for explicitly registered RobotSim robot models."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "assets/robots/registry.json"
HASHES_PATH = ROOT / "assets/robots/source_hashes.json"
COMPILED_PATH = ROOT / "assets/robots/compiled_model_signatures.json"
STRUCTURE_FIELDS = ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "nmesh", "nsite", "neq")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_hash(
    registry: dict[str, Any], source_hashes: dict[str, Any], compiled: dict[str, Any]
) -> str:
    payload = {"registry": registry, "source_hashes": source_hashes, "compiled": compiled}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def find_asset(registry: dict[str, Any], asset_id: str) -> dict[str, Any]:
    for asset in registry.get("assets", []):
        if asset.get("asset_id") == asset_id:
            return asset
    raise KeyError(f"unknown asset_id: {asset_id}")


def actual_source_revision(root: Path, declared_revision: str | None) -> tuple[str | None, str]:
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if git.returncode == 0:
        return git.stdout.strip(), "git"
    return declared_revision, "explicit source snapshot revision"


def expected_profile_files(profile: dict[str, Any], root: Path) -> set[str]:
    files = {str(item).replace("\\", "/") for item in profile.get("hash_files", [])}
    for relative_root in profile.get("hash_roots", []):
        directory = root / relative_root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
    return files


def build_profile_hashes(profile: dict[str, Any], root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in sorted(expected_profile_files(profile, root)):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"source file missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def check_profile(
    profile_id: str,
    profile: dict[str, Any],
    recorded: dict[str, Any],
    root: Path,
    declared_revision: str | None = None,
) -> list[str]:
    errors: list[str] = []
    revision, revision_source = actual_source_revision(root, declared_revision)
    expected_revision = profile["commit"]
    if revision != expected_revision:
        errors.append(
            f"source revision mismatch for {profile_id}: expected {expected_revision}, "
            f"got {revision or 'unknown'} ({revision_source})"
        )

    locked = recorded.get("profiles", {}).get(profile_id)
    if not locked:
        return errors + [f"source hash profile missing from {HASHES_PATH}: {profile_id}"]
    if locked.get("commit") != expected_revision:
        errors.append(
            f"recorded source profile commit mismatch for {profile_id}: "
            f"expected {expected_revision}, got {locked.get('commit')}"
        )

    expected_paths = expected_profile_files(profile, root)
    locked_files = locked.get("files", {})
    for relative in sorted(expected_paths):
        path = root / relative
        expected_hash = locked_files.get(relative)
        if not path.is_file():
            errors.append(f"missing source model/mesh/license file: {relative}")
            continue
        if expected_hash is None:
            errors.append(f"unregistered source file inside hashed profile: {relative}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative}: expected {expected_hash}, got {actual_hash}")
    for extra in sorted(set(locked_files) - expected_paths):
        path = root / extra
        if path.is_file():
            actual_hash = sha256_file(path)
            if actual_hash != locked_files[extra]:
                errors.append(f"SHA-256 mismatch: {extra}: expected {locked_files[extra]}, got {actual_hash}")
        else:
            errors.append(f"recorded source file missing: {extra}")
    return errors


def mujoco_snapshot(model_path: Path) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo Python package is required for compiled-model validation") from exc

    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)]
    mesh_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) or "" for i in range(model.nmesh)]
    body_names_by_id = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    joint_records = []
    for i, name in enumerate(joint_names):
        joint_records.append({
            "name": name,
            "type": int(model.jnt_type[i]),
            "body": body_names_by_id[int(model.jnt_bodyid[i])],
            "limited": bool(model.jnt_limited[i]),
            "range": model.jnt_range[i].tolist() if model.jnt_limited[i] else None,
            "axis": model.jnt_axis[i].tolist(),
            "pos": model.jnt_pos[i].tolist(),
        })
    geom_records = []
    contact_geom_names = []
    for i, name in enumerate(geom_names):
        contact_enabled = bool(model.geom_contype[i] or model.geom_conaffinity[i])
        if contact_enabled:
            contact_geom_names.append(name)
        data_id = int(model.geom_dataid[i])
        mesh_name = mesh_names[data_id] if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and data_id >= 0 else None
        geom_records.append({
            "name": name,
            "body": body_names_by_id[int(model.geom_bodyid[i])],
            "type": int(model.geom_type[i]),
            "contype": int(model.geom_contype[i]),
            "conaffinity": int(model.geom_conaffinity[i]),
            "group": int(model.geom_group[i]),
            "mesh": mesh_name,
            "pos": model.geom_pos[i].tolist(),
            "size": model.geom_size[i].tolist(),
        })
    actuator_records = []
    for i, name in enumerate(actuator_names):
        actuator_records.append({
            "name": name,
            "transmission_type": int(model.actuator_trntype[i]),
            "transmission_ids": model.actuator_trnid[i].tolist(),
            "control_limited": bool(model.actuator_ctrllimited[i]),
            "control_range": model.actuator_ctrlrange[i].tolist() if model.actuator_ctrllimited[i] else None,
            "force_limited": bool(model.actuator_forcelimited[i]),
            "force_range": model.actuator_forcerange[i].tolist() if model.actuator_forcelimited[i] else None,
        })
    geom_group_counts: dict[str, int] = {}
    for group in model.geom_group:
        key = str(int(group))
        geom_group_counts[key] = geom_group_counts.get(key, 0) + 1
    return {
        "mujoco_version": mujoco.__version__,
        "signature": {
            "nq": model.nq, "nv": model.nv, "nu": model.nu, "njnt": model.njnt,
            "nbody": model.nbody, "ngeom": model.ngeom, "nmesh": model.nmesh,
            "nsite": model.nsite, "neq": model.neq,
        },
        "joint_names": joint_names,
        "body_names": body_names,
        "geom_names": geom_names,
        "contact_geom_names": contact_geom_names,
        "actuator_names": actuator_names,
        "mesh_names": mesh_names,
        "joint_records": joint_records,
        "geom_records": geom_records,
        "actuator_records": actuator_records,
        "geom_group_counts": geom_group_counts,
    }


def compare_model(
    asset: dict[str, Any], snapshot: dict[str, Any], compiled_profile: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    expected = asset.get("expected_structure", {})
    actual = snapshot["signature"]
    for field in STRUCTURE_FIELDS:
        if field in expected and actual.get(field) != expected[field]:
            errors.append(f"compiled {field} mismatch: expected {expected[field]}, got {actual.get(field)}")

    groups = (
        snapshot["joint_names"] + snapshot["body_names"] + snapshot["geom_names"] + snapshot["actuator_names"]
    )
    all_names = "\n".join(groups)
    for token in asset.get("required_joint_tokens", []):
        if token not in all_names:
            errors.append(f"required model name token missing: {token}")
    for token in asset.get("forbidden_joint_tokens", []):
        if token in all_names:
            errors.append(f"forbidden/mutually-exclusive model name token present: {token}")
    if compiled_profile:
        if snapshot["mujoco_version"] != compiled_profile.get("mujoco_version"):
            errors.append(
                f"MuJoCo version mismatch: expected {compiled_profile.get('mujoco_version')}, "
                f"got {snapshot['mujoco_version']}"
            )
        expected_snapshot = compiled_profile.get("snapshot", {})
        for field in ("joint_names", "body_names", "geom_names", "actuator_names", "mesh_names"):
            missing = sorted(set(expected_snapshot.get(field, [])) - set(snapshot.get(field, [])))
            extra = sorted(set(snapshot.get(field, [])) - set(expected_snapshot.get(field, [])))
            if missing:
                errors.append(f"compiled {field} missing exact names: {', '.join(missing)}")
            if extra:
                errors.append(f"compiled {field} has unexpected exact names: {', '.join(extra)}")
        expected_snapshot = compiled_profile.get("snapshot", {})
        for field in ("joint_records", "geom_records", "actuator_records", "contact_geom_names", "geom_group_counts"):
            expected_value = expected_snapshot.get(field)
            actual_value = snapshot.get(field)
            if expected_value == actual_value:
                continue
            if field in ("joint_records", "geom_records", "actuator_records"):
                key_name = "name"
                expected_by_name = {row[key_name]: row for row in expected_value or []}
                actual_by_name = {row[key_name]: row for row in actual_value or []}
                for missing in sorted(set(expected_by_name) - set(actual_by_name)):
                    errors.append(f"compiled {field} missing: {missing}")
                for extra in sorted(set(actual_by_name) - set(expected_by_name)):
                    errors.append(f"compiled {field} unexpected: {extra}")
                for name in sorted(set(expected_by_name) & set(actual_by_name)):
                    if expected_by_name[name] != actual_by_name[name]:
                        errors.append(
                            f"compiled {field} mismatch for {name}: expected {expected_by_name[name]}, "
                            f"got {actual_by_name[name]}"
                        )
            else:
                errors.append(f"compiled {field} mismatch: expected {expected_value}, got {actual_value}")
    return errors


def command_record_profile(args: argparse.Namespace, registry: dict[str, Any]) -> int:
    profile_id = args.source_profile
    try:
        profile = registry["source_profiles"][profile_id]
    except KeyError:
        print(f"unknown source profile: {profile_id}", file=sys.stderr)
        return 2
    root = Path(args.source_root).resolve()
    revision, _ = actual_source_revision(root, args.source_revision)
    if revision != profile["commit"]:
        print(
            f"refusing to record source hashes: expected revision {profile['commit']}, got {revision}",
            file=sys.stderr,
        )
        return 2
    try:
        files = build_profile_hashes(profile, root)
    except (FileNotFoundError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    current = json.loads(HASHES_PATH.read_text(encoding="utf-8")) if HASHES_PATH.exists() else {"schema_version": 1, "profiles": {}}
    current.setdefault("profiles", {})[profile_id] = {"commit": profile["commit"], "files": files}
    HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASHES_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {len(files)} files for {profile_id} at {profile['commit']}")
    return 0


def command_record_compiled(args: argparse.Namespace, registry: dict[str, Any]) -> int:
    try:
        asset = find_asset(registry, args.asset_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    source_profile_id = asset["source_profile"]
    source_profile = registry["source_profiles"][source_profile_id]
    source_hashes = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
    root = Path(args.source_root).resolve()
    errors = check_profile(
        source_profile_id, source_profile, source_hashes, root, args.source_revision
    )
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    model_path = root / asset["model_path"]
    expected_hash = source_hashes["profiles"][source_profile_id]["files"].get(asset["model_path"])
    if not expected_hash or sha256_file(model_path) != expected_hash:
        print(f"source model hash mismatch: {asset['model_path']}", file=sys.stderr)
        return 1
    try:
        snapshot = mujoco_snapshot(model_path)
    except Exception as exc:
        print(f"MuJoCo compile failed for {model_path}: {exc}", file=sys.stderr)
        return 1
    errors = compare_model(asset, snapshot)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    compiled = json.loads(COMPILED_PATH.read_text(encoding="utf-8")) if COMPILED_PATH.exists() else {"schema_version": 1, "profiles": {}}
    compiled.setdefault("profiles", {})[args.asset_id] = {
        "source_profile": source_profile_id,
        "commit": source_profile["commit"],
        "model_path": asset["model_path"],
        "model_sha256": expected_hash,
        "mujoco_version": snapshot["mujoco_version"],
        "snapshot": snapshot,
    }
    COMPILED_PATH.write_text(json.dumps(compiled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded compiled signature for {args.asset_id} with MuJoCo {snapshot['mujoco_version']}")
    return 0


def command_validate(args: argparse.Namespace, registry: dict[str, Any]) -> int:
    hashes = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
    compiled = json.loads(COMPILED_PATH.read_text(encoding="utf-8"))
    actual_manifest_hash = canonical_manifest_hash(registry, hashes, compiled)
    if args.manifest_sha256 != actual_manifest_hash:
        print(
            f"manifest SHA-256 mismatch: expected {actual_manifest_hash}, got {args.manifest_sha256}",
            file=sys.stderr,
        )
        return 2
    try:
        asset = find_asset(registry, args.asset_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    errors: list[str] = []
    profile_id = asset["source_profile"]
    profile = registry["source_profiles"][profile_id]
    if args.source_root:
        errors.extend(
            check_profile(profile_id, profile, hashes, Path(args.source_root).resolve(), args.source_revision)
        )
    elif args.source_revision:
        print("--source-revision requires --source-root", file=sys.stderr)
        return 2

    locked = hashes.get("profiles", {}).get(profile_id, {}).get("files", {})
    root_model_hash = locked.get(asset["model_path"])
    if not root_model_hash:
        errors.append(f"root model is not covered by source hash manifest: {asset['model_path']}")
    if args.model:
        model_path = Path(args.model).resolve()
        if not model_path.is_file():
            errors.append(f"runtime model file missing: {model_path}")
        else:
            runtime_hash = sha256_file(model_path)
            if runtime_hash != root_model_hash:
                errors.append(
                    f"runtime root-model SHA-256 mismatch: expected {root_model_hash}, got {runtime_hash} ({model_path})"
                )
            try:
                compiled_profile = compiled.get("profiles", {}).get(args.asset_id)
                if not compiled_profile:
                    errors.append(f"compiled model signature missing for asset_id: {args.asset_id}")
                elif compiled_profile.get("model_sha256") != root_model_hash:
                    errors.append(f"compiled signature source hash is stale for asset_id: {args.asset_id}")
                else:
                    errors.extend(compare_model(asset, mujoco_snapshot(model_path), compiled_profile))
            except Exception as exc:
                errors.append(f"MuJoCo compile failed for {model_path}: {exc}")
    elif not args.source_root:
        print("provide --model and/or --source-root", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {args.asset_id}; manifest={actual_manifest_hash}; source={profile['commit']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest-hash", help="print combined identity-manifest SHA-256")
    record = subparsers.add_parser("record-source-profile", help="record hashes from an exact pinned source")
    record.add_argument("--source-profile", required=True)
    record.add_argument("--source-root", required=True)
    record.add_argument("--source-revision", help="required assertion for a raw snapshot without Git metadata")
    record_compiled = subparsers.add_parser("record-compiled-signature", help="record exact names from a pinned source model")
    record_compiled.add_argument("--asset-id", required=True)
    record_compiled.add_argument("--source-root", required=True)
    record_compiled.add_argument("--source-revision", help="required assertion for a raw snapshot without Git metadata")
    validate = subparsers.add_parser("validate", help="validate source and/or compiled MuJoCo identity")
    validate.add_argument("--asset-id", required=True)
    validate.add_argument("--manifest-sha256", required=True)
    validate.add_argument("--source-root")
    validate.add_argument("--source-revision")
    validate.add_argument("--model")
    args = parser.parse_args()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if args.command == "manifest-hash":
        hashes = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
        compiled = json.loads(COMPILED_PATH.read_text(encoding="utf-8"))
        print(canonical_manifest_hash(registry, hashes, compiled))
        return 0
    if args.command == "record-source-profile":
        return command_record_profile(args, registry)
    if args.command == "record-compiled-signature":
        return command_record_compiled(args, registry)
    return command_validate(args, registry)


if __name__ == "__main__":
    raise SystemExit(main())
