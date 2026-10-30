#!/usr/bin/env python3
"""Open the bilateral RAICOM+STL mount audit in the native MuJoCo slider GUI."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.audit.is_file():
        raise SystemExit("mount-audit result is missing")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    required = ("OFFICIAL_WRIST_PARENT", "OFFICIAL_FIXED_MOUNT_ORIGINS", "BILATERAL_STL_STATIC_MJ_FORWARD")
    if any(audit.get("gates", {}).get(gate) != "PASS" for gate in required):
        raise SystemExit("mount audit has not passed source/static gates")
    overlay = Path(audit["generated_overlay"]["path"])
    if not overlay.is_file() or sha256(overlay) != audit["generated_overlay"]["sha256"]:
        raise SystemExit("mount overlay is missing or changed")
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    if loaded.model_sha256 != audit["stl_model"]["model_sha256"]:
        raise SystemExit("registered STL model identity differs from mount audit")
    model = mujoco.MjModel.from_xml_path(str(overlay))
    trajectory = load_registered_trajectory(loaded.package_path)
    print(json.dumps({"viewer": "MuJoCo bilateral STL static review; 0 mj_step", "overlay": str(overlay), "mounts": audit["official_mounts"], "controls": "A/D left, Left/Right right, V/C/O render mode, mouse orbit/pan/zoom"}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    viewer = load_module("stl_viewer", REPO / "scripts/research/issue46/scene/full_x2_stl_review.py")
    viewer.run_live_viewer(model, trajectory)


if __name__ == "__main__":
    main()
