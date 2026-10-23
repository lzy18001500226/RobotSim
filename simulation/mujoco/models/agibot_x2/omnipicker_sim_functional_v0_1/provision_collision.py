"""Explicitly provision hash-locked CoACD pieces from the pinned X2 source."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any

import numpy as np


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[4]
SOURCE_MANIFEST = PACKAGE / "source_assets.json"
PROXY_MANIFEST = PACKAGE / "collision_proxy_manifest.json"
OUTPUT = PACKAGE / "collision_generated"
EXPECTED_COACD = "1.0.14"
EXPECTED_NUMPY = "2.2.6"


class ProvisionError(RuntimeError):
    """A pinned source or generated collision proxy did not match its lock."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvisionError("Git is required to validate the pinned vendor source") from exc
    if result.returncode != 0:
        raise ProvisionError("could not validate pinned vendor checkout metadata")
    return result.stdout.strip()


def source_root() -> tuple[Path, dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    root = REPOSITORY / manifest["repository_path"]
    if not root.is_dir():
        raise ProvisionError(
            f"missing pinned vendor checkout: {root}; provision it using docs/setup/reference-sources.md"
        )
    if git(root, "remote", "get-url", "origin") != manifest["repository"]:
        raise ProvisionError("vendor checkout origin does not match source_assets.json")
    if git(root, "rev-parse", "HEAD") != manifest["commit"]:
        raise ProvisionError("vendor checkout HEAD does not match the pinned commit")
    if git(root, "status", "--porcelain"):
        raise ProvisionError("vendor checkout is dirty; refusing to generate collision assets")
    required = manifest["required_assets"]
    assets = [required["urdf"], *required["meshes"]]
    for asset in assets:
        path = root / asset["path"]
        if not path.is_file():
            raise ProvisionError(f"missing required vendor asset: {asset['path']}")
        if sha256(path) != asset["sha256"]:
            raise ProvisionError(f"vendor asset SHA-256 mismatch: {asset['path']}")
    return root, manifest


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    content = path.read_bytes()
    if len(content) < 84:
        raise ProvisionError(f"not a binary STL: {path.name}")
    triangle_count = struct.unpack_from("<I", content, 80)[0]
    if len(content) != 84 + 50 * triangle_count:
        raise ProvisionError(f"malformed or non-binary STL: {path.name}")
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    triangles = np.frombuffer(content[84:], dtype=dtype)["vertices"].astype(np.float32)
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices.astype(np.float32), inverse.reshape(-1, 3).astype(np.uint32)


def write_binary_stl(vertices: np.ndarray, faces: np.ndarray, name: str) -> bytes:
    triangles = np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-20
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    output = np.zeros(len(triangles), dtype=dtype)
    output["normal"] = normals
    output["vertices"] = triangles
    header = ("RobotSim SIMULATION_ONLY CoACD source-derived convex piece " + name).encode()[:80].ljust(80, b" ")
    return header + struct.pack("<I", len(triangles)) + output.tobytes()


def provision() -> dict[str, Any]:
    try:
        coacd_version = importlib.metadata.version("coacd")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProvisionError("install coacd==1.0.14 in the isolated model environment") from exc
    if coacd_version != EXPECTED_COACD:
        raise ProvisionError(f"CoACD version mismatch: expected {EXPECTED_COACD}, got {coacd_version}")
    if np.__version__ != EXPECTED_NUMPY:
        raise ProvisionError(f"NumPy version mismatch: expected {EXPECTED_NUMPY}, got {np.__version__}")
    try:
        import coacd
    except ImportError as exc:
        raise ProvisionError("install coacd==1.0.14 in the isolated model environment") from exc

    source, source_manifest = source_root()
    proxy_manifest = json.loads(PROXY_MANIFEST.read_text(encoding="utf-8"))
    if proxy_manifest["version"] != EXPECTED_COACD:
        raise ProvisionError("collision proxy manifest and installed CoACD version differ")
    settings = proxy_manifest["settings"]
    expected: dict[str, dict[str, Any]] = {}
    for mesh in proxy_manifest["mesh_decompositions"]:
        source_mesh = source / mesh["source_mesh"]
        if not source_mesh.is_file() or sha256(source_mesh) != mesh["source_sha256"]:
            raise ProvisionError(f"collision source mesh hash mismatch: {mesh['source_mesh']}")
        for piece in mesh["parts"]:
            expected[piece["filename"]] = piece
    if len(expected) != 161:
        raise ProvisionError(f"expected 161 collision-piece hashes, found {len(expected)}")

    # Reuse an existing generated directory only if it is already complete and exact.
    if OUTPUT.exists():
        existing = {p.name for p in OUTPUT.iterdir() if p.is_file()}
        if existing != set(expected):
            raise ProvisionError("collision_generated exists but is incomplete or contains unregistered files")
        mismatches = [name for name, record in expected.items() if sha256(OUTPUT / name) != record["sha256"]]
        if mismatches:
            raise ProvisionError("existing collision proxy SHA-256 mismatch: " + ", ".join(mismatches[:5]))
        return {
            "status": "REUSED_HASH_VERIFIED",
            "source_commit": source_manifest["commit"],
            "coacd_version": coacd_version,
            "piece_count": len(expected),
            "output": str(OUTPUT),
        }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    written: list[str] = []
    for mesh in proxy_manifest["mesh_decompositions"]:
        source_mesh = source / mesh["source_mesh"]
        content = source_mesh.read_bytes()
        triangle_count = struct.unpack_from("<I", content, 80)[0]
        dtype = np.dtype(
            [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
        )
        triangles = np.frombuffer(content[84:], dtype=dtype)["vertices"].astype(np.float32)
        vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3).astype(np.uint32)
        if len(vertices) != mesh["vertices_unique"] or triangle_count != mesh["triangles"]:
            raise ProvisionError(f"source STL topology differs from collision manifest: {mesh['source_mesh']}")
        parts = coacd.run_coacd(coacd.Mesh(vertices.astype(np.float32), faces), **settings)
        if len(parts) != len(mesh["parts"]):
            raise ProvisionError(
                f"CoACD piece count differs for {mesh['source_mesh']}: expected {len(mesh['parts'])}, got {len(parts)}"
            )
        for index, (piece_vertices, piece_faces) in enumerate(parts):
            name = f"{Path(mesh['source_mesh']).stem}_coacd_{index:02d}.stl"
            record = expected[name]
            data = write_binary_stl(piece_vertices, piece_faces, record["name"])
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ProvisionError(f"generated CoACD piece hash differs from frozen reference: {name}")
            target = OUTPUT / name
            with target.open("xb") as stream:
                stream.write(data)
            written.append(name)
    if set(written) != set(expected):
        raise ProvisionError("generated collision piece set is incomplete")
    return {
        "status": "GENERATED_AND_HASH_VERIFIED",
        "source_commit": source_manifest["commit"],
        "coacd_version": coacd_version,
        "numpy_version": np.__version__,
        "piece_count": len(written),
        "output": str(OUTPUT),
    }


if __name__ == "__main__":
    print(json.dumps(provision(), indent=2, sort_keys=True))
