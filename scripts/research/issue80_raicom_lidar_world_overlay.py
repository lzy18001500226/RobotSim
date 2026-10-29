#!/usr/bin/env python3
"""Build an evidence-local RayCaster overlay that records local and world rays.

The generated bundle keeps the accepted X2 LiDAR scene/YAML overlay unchanged
and replaces only the custom sensor plugin binary.  Its additional world_xyz
trace field is explicitly SIMULATION_ONLY MuJoCo geometry evidence; it is not
a hardware LiDAR localization transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from issue80_raicom_mjdata_capture import HEADER_SHA256


PLUGIN_SOURCE = Path(__file__).with_name("issue80_raicom_lidar_plugin.cc")
RUNTIME_LIB = Path(
    "/home/lzy18001500226/projects/RobotSim/"
    "simulation/mujoco/raicom2026/sim_mujoco/lib"
)


class WorldOverlayError(RuntimeError):
    """The evidence-local source/world-ray overlay is incomplete."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_headers(root: Path) -> None:
    for name, expected in HEADER_SHA256.items():
        path = root / "mujoco" / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise WorldOverlayError(f"MuJoCo 3.3.7 header is missing or unverified: {name}")


def build(
    output_dir: Path, *, base_overlay_source: Path, headers_include_root: Path,
) -> dict:
    output = output_dir.resolve(strict=True)
    allowed = {".robotsim-output-reservation.json"}
    if output.is_symlink() or any(item.name not in allowed for item in output.iterdir()):
        raise WorldOverlayError("output directory must contain only its governed reservation")
    source = base_overlay_source.resolve(strict=True)
    overlay = source / "lidar_overlay"
    plugins = source / "mujoco_plugin"
    if source.is_symlink() or overlay.is_symlink() or plugins.is_symlink() or not overlay.is_dir() or not plugins.is_dir():
        raise WorldOverlayError("base LiDAR overlay source is missing or unsafe")
    headers = headers_include_root.resolve(strict=True)
    verify_headers(headers)
    if PLUGIN_SOURCE.is_symlink() or not PLUGIN_SOURCE.is_file():
        raise WorldOverlayError("world-ray plugin source is missing or unsafe")
    if not RUNTIME_LIB.is_dir() or not (RUNTIME_LIB / "libmujoco.so.3.3.7").is_file():
        raise WorldOverlayError("pinned RAICOM MuJoCo runtime library is unavailable")
    output_overlay = output / "lidar_overlay"
    output_plugins = output / "mujoco_plugin"
    shutil.copytree(overlay, output_overlay)
    shutil.copytree(plugins, output_plugins)
    copied_source = output / PLUGIN_SOURCE.name
    shutil.copyfile(PLUGIN_SOURCE, copied_source)
    library = output_plugins / "libissue80_ray.so"
    command = [
        "g++", "-shared", "-fPIC", "-O2", "-std=c++17", "-Wall", "-Wextra",
        "-Werror=return-type", "-I", str(headers), str(copied_source),
        "-L", str(RUNTIME_LIB), "-Wl,-rpath," + str(RUNTIME_LIB), "-lmujoco",
        "-o", str(library),
    ]
    built = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if built.returncode != 0 or library.is_symlink() or not library.is_file():
        raise WorldOverlayError(
            "world-ray plugin compilation failed: " + json.dumps(
                {"returncode": built.returncode, "stdout": built.stdout[-4000:], "stderr": built.stderr[-4000:]},
                sort_keys=True,
            )
        )
    dynamic = subprocess.run(["readelf", "-d", str(library)], text=True, capture_output=True, timeout=10)
    if dynamic.returncode != 0 or "libmujoco.so.3.3.7" not in dynamic.stdout:
        raise WorldOverlayError("world-ray plugin does not link the pinned MuJoCo 3.3.7 runtime")
    manifest = {
        "schema": "robotsim.issue80.raicom-lidar-world-trace-overlay.v1",
        "classification": "SIMULATION_ONLY_WORLD_PROJECTED_RAY_TRACE_OVERLAY",
        "base_overlay_source": str(source),
        "base_overlay_sha256": {
            path.name: sha256(path) for path in sorted(overlay.iterdir()) if path.is_file()
        },
        "plugin_source": {"path": str(PLUGIN_SOURCE), "sha256": sha256(PLUGIN_SOURCE)},
        "headers_include_root": str(headers),
        "headers_sha256": HEADER_SHA256,
        "runtime_mujoco_library": {
            "path": str(RUNTIME_LIB / "libmujoco.so.3.3.7"),
            "sha256": sha256(RUNTIME_LIB / "libmujoco.so.3.3.7"),
        },
        "build_command": command,
        "compile_stdout": built.stdout[-4000:],
        "compile_stderr": built.stderr[-4000:],
        "plugin_sha256": sha256(library),
        "dynamic_section": dynamic.stdout,
        "unchanged": ["X2 body", "X2 joint", "X2 actuator", "X2 collision geometry", "MC binaries/config"],
        "added_trace_field": "world_xyz = MuJoCo camera-origin plus world ray direction times measured hit",
        "limitation": "world_xyz is simulator geometry ground truth and must never be described as hardware LiDAR localization.",
    }
    (output / "LIDAR_WORLD_TRACE_OVERLAY_PROVENANCE.json").write_text(
        json.dumps(manifest, indent=2) + chr(10), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-overlay-source", type=Path, required=True)
    parser.add_argument("--headers-include-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(
            args.output_dir,
            base_overlay_source=args.base_overlay_source,
            headers_include_root=args.headers_include_root,
        ), indent=2))
    except (OSError, WorldOverlayError, subprocess.SubprocessError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
