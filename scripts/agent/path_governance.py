"""Canonical, no-overwrite output paths for local RobotSim evidence."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


CANONICAL_REPO_ROOT = "/home/lzy18001500226/projects/RobotSim"
WINDOWS_DISTRO = "Ubuntu-22.04"
_WINDOWS_UNC = re.compile(
    r"^[\\/]{2}wsl\.localhost[\\/]+Ubuntu-22\.04(?:[\\/]+(?P<tail>.*))?$",
    re.IGNORECASE,
)
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_SAFE_RUN_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_WINDOWS_INVALID = set('<>:"|?*')


class PathGovernanceError(ValueError):
    """An output path violates RobotSim's local evidence policy."""


@dataclass(frozen=True)
class ArtifactLocation:
    wsl_path: str
    windows_unc: str
    open_command: str
    native_path: Path = field(repr=False, compare=False)
    approved_root: str = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, str]:
        return {
            "WSL_PATH": self.wsl_path,
            "WINDOWS_UNC": self.windows_unc,
            "OPEN_COMMAND": self.open_command,
        }


def detect_environment(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    os_release: str | None = None,
) -> str:
    """Return ``windows``, ``wsl``, or ``linux`` for the current process."""
    env = os.environ if environ is None else environ
    if platform_name in {"nt", "win32", "windows"} or (platform_name is None and os.name == "nt"):
        return "windows"
    if platform_name not in {None, "posix", "linux", "wsl"}:
        raise PathGovernanceError("unsupported operating system")
    release = os_release
    if release is None:
        try:
            release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
        except OSError:
            release = ""
    if platform_name == "wsl" or env.get("WSL_INTEROP") or env.get("WSL_DISTRO_NAME") or "microsoft" in release.casefold():
        return "wsl"
    return "linux"


def _unc_tail_to_wsl_path(value: str) -> str | None:
    match = _WINDOWS_UNC.fullmatch(value)
    if match is None:
        return None
    tail = match.group("tail") or ""
    components = tail.replace("/", "\\").split("\\") if tail else []
    if any(part in {"", ".", ".."} for part in components):
        raise PathGovernanceError("Windows UNC path contains an invalid component")
    return "/" + "/".join(components)


def _validate_posix_lexical_path(value: str) -> str:
    if not value.startswith("/") or value.startswith("//"):
        raise PathGovernanceError("output path must be an absolute canonical path")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise PathGovernanceError("output path contains a control character")
    if any(part in {".", ".."} for part in value.split("/")):
        raise PathGovernanceError("output path traversal is not allowed")
    normalized = str(PurePosixPath(value))
    if normalized == "/":
        raise PathGovernanceError("filesystem root is not an output path")
    return normalized


def _to_wsl_path(value: str | os.PathLike[str], environment: str) -> str:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw:
        raise PathGovernanceError("output path must be a non-empty string")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in raw):
        raise PathGovernanceError("output path contains a control character")

    if environment == "windows":
        if raw.startswith("/") and not raw.startswith("//wsl.localhost/"):
            raise PathGovernanceError("Linux /home or /mnt paths are invalid in native Windows processes")
        unc_path = _unc_tail_to_wsl_path(raw)
        if unc_path is None:
            if _WINDOWS_DRIVE.match(raw):
                raise PathGovernanceError("drive-letter paths are not canonical RobotSim evidence paths")
            raise PathGovernanceError("native Windows output must use the Ubuntu-22.04 WSL UNC path")
        return _validate_posix_lexical_path(unc_path)

    unc_path = _unc_tail_to_wsl_path(raw)
    if unc_path is not None:
        return _validate_posix_lexical_path(unc_path)
    if raw.startswith("//"):
        raise PathGovernanceError("ambiguous network paths are not canonical evidence paths")
    if _WINDOWS_DRIVE.match(raw):
        raise PathGovernanceError("Windows drive paths are invalid in Linux/WSL processes")
    if "\\" in raw:
        raise PathGovernanceError("Windows path syntax is not valid for a Linux/WSL process")
    if raw.casefold() == "/mnt/c" or raw.casefold().startswith("/mnt/c/"):
        raise PathGovernanceError("Windows-mounted paths are not approved RobotSim evidence paths")
    return _validate_posix_lexical_path(raw)


def _is_below(path: str, root: str) -> bool:
    try:
        PurePosixPath(path).relative_to(PurePosixPath(root))
        return True
    except ValueError:
        return False


def _windows_unc_for_wsl_path(wsl_path: str) -> str:
    parts = PurePosixPath(wsl_path).parts[1:]
    if not parts or any(
        any(ord(char) < 0x20 or ord(char) == 0x7F or char in _WINDOWS_INVALID for char in part)
        or part.endswith((" ", "."))
        for part in parts
    ):
        raise PathGovernanceError("output path cannot be represented as a valid Windows UNC path")
    return "\\\\wsl.localhost\\" + WINDOWS_DISTRO + "\\" + "\\".join(parts)


def _open_command(windows_unc: str) -> str:
    quoted_path = windows_unc.replace("'", "''")
    return f"Start-Process -FilePath 'explorer.exe' -ArgumentList @('{quoted_path}')"


def normalize_output_path(
    path: str | os.PathLike[str],
    *,
    environment: str | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> str:
    """Normalize a path to its WSL spelling without touching the filesystem."""
    runtime = environment or detect_environment(platform_name=platform_name, environ=environ)
    if runtime not in {"windows", "wsl", "linux"}:
        raise PathGovernanceError("environment must be windows, wsl, or linux")
    return _to_wsl_path(path, runtime)


def _native_path(wsl_path: str, environment: str) -> Path:
    if environment == "windows":
        return Path(_windows_unc_for_wsl_path(wsl_path))
    return Path(wsl_path)


def _same_path(first: Path, second: Path, environment: str) -> bool:
    a, b = os.path.normpath(str(first)), os.path.normpath(str(second))
    return a.casefold() == b.casefold() if environment == "windows" else a == b


def _is_physically_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_output_path(
    path: str | os.PathLike[str],
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    require_exists: bool = False,
    require_new: bool = False,
    expected_kind: str | None = None,
    environment: str | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> ArtifactLocation:
    """Validate policy and resolved containment before an output is created."""
    runtime = environment or detect_environment(platform_name=platform_name, environ=environ)
    if runtime not in {"windows", "wsl", "linux"}:
        raise PathGovernanceError("environment must be windows, wsl, or linux")
    wsl_path = _to_wsl_path(path, runtime)
    reviews_root = str(PurePosixPath(CANONICAL_REPO_ROOT) / "reviews")
    approved_root = reviews_root if _is_below(wsl_path, reviews_root) else ""

    for authorized in authorized_roots:
        authorized_path = _to_wsl_path(authorized, runtime)
        if _is_below(wsl_path, authorized_path):
            approved_root = authorized_path
            break
    if not approved_root:
        raise PathGovernanceError("output path is outside the approved RobotSim reviews root")

    native = _native_path(wsl_path, runtime)
    root_native = _native_path(approved_root, runtime)
    if not root_native.exists() or not root_native.is_dir():
        raise PathGovernanceError("approved output root is unavailable")
    if root_native.is_symlink() or (
        hasattr(root_native, "is_junction") and root_native.is_junction()
    ):
        raise PathGovernanceError("approved output root must not be a symlink or junction")
    try:
        resolved_root = root_native.resolve(strict=True)
        resolved_target = native.resolve(strict=False)
    except OSError as exc:
        raise PathGovernanceError("output path cannot be resolved safely") from exc
    if not _same_path(root_native.absolute(), resolved_root, runtime):
        raise PathGovernanceError("approved output root resolves through a symlink or junction")
    if not _is_physically_below(resolved_target, resolved_root):
        raise PathGovernanceError("resolved output path escapes its approved root")
    if require_new and native.exists():
        raise PathGovernanceError("output already exists; refusing to overwrite previous evidence")
    if require_exists and not native.exists():
        raise PathGovernanceError("advertised output does not exist")
    if expected_kind == "file" and (not native.is_file() or native.is_symlink()):
        raise PathGovernanceError("advertised output is not a regular file")
    if expected_kind == "directory" and (not native.is_dir() or native.is_symlink()):
        raise PathGovernanceError("output is not a regular directory")
    if expected_kind not in {None, "file", "directory"}:
        raise PathGovernanceError("expected_kind must be file or directory")

    windows_unc = _windows_unc_for_wsl_path(wsl_path)
    return ArtifactLocation(wsl_path, windows_unc, _open_command(windows_unc), native, approved_root)


def describe_output_path(
    path: str | os.PathLike[str],
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    environment: str | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> dict[str, str]:
    return validate_output_path(
        path,
        authorized_roots=authorized_roots,
        environment=environment,
        environ=environ,
        platform_name=platform_name,
    ).as_dict()


def create_output_directory(
    path: str | os.PathLike[str],
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    environment: str | None = None,
) -> ArtifactLocation:
    authorized = tuple(authorized_roots)
    location = validate_output_path(
        path, authorized_roots=authorized, require_new=True, environment=environment
    )
    try:
        location.native_path.parent.mkdir(parents=True, exist_ok=True)
        location.native_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PathGovernanceError("output already exists; refusing to overwrite previous evidence") from exc
    return validate_output_path(
        location.wsl_path,
        authorized_roots=authorized,
        require_exists=True,
        expected_kind="directory",
        environment=environment,
    )


def create_unique_output_directory(
    parent: str | os.PathLike[str],
    *,
    prefix: str,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    environment: str | None = None,
    now: datetime | None = None,
) -> ArtifactLocation:
    if not _SAFE_RUN_PREFIX.fullmatch(prefix):
        raise PathGovernanceError("run directory prefix must be a short safe identifier")
    authorized = tuple(authorized_roots)
    base = validate_output_path(
        parent,
        authorized_roots=authorized,
        require_exists=True,
        expected_kind="directory",
        environment=environment,
    )
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"-{attempt:02d}"
        candidate = f"{base.wsl_path}/{prefix}-{timestamp}{suffix}"
        try:
            return create_output_directory(
                candidate, authorized_roots=authorized, environment=environment
            )
        except PathGovernanceError as exc:
            if "already exists" not in str(exc):
                raise
    raise PathGovernanceError("could not allocate a unique output directory")


def write_new_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    environment: str | None = None,
) -> ArtifactLocation:
    location = validate_output_path(
        path, authorized_roots=authorized_roots, require_new=True, environment=environment
    )
    if not location.native_path.parent.is_dir():
        raise PathGovernanceError("reserve the output run directory before writing artifacts")
    try:
        with location.native_path.open("x", encoding=encoding, newline="") as stream:
            stream.write(text)
    except FileExistsError as exc:
        raise PathGovernanceError("output already exists; refusing to overwrite previous evidence") from exc
    return validate_output_path(
        location.wsl_path,
        authorized_roots=authorized_roots,
        require_exists=True,
        expected_kind="file",
        environment=environment,
    )


def write_new_bytes(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    environment: str | None = None,
) -> ArtifactLocation:
    location = validate_output_path(
        path, authorized_roots=authorized_roots, require_new=True, environment=environment
    )
    if not location.native_path.parent.is_dir():
        raise PathGovernanceError("reserve the output run directory before writing artifacts")
    try:
        with location.native_path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise PathGovernanceError("output already exists; refusing to overwrite previous evidence") from exc
    return validate_output_path(
        location.wsl_path,
        authorized_roots=authorized_roots,
        require_exists=True,
        expected_kind="file",
        environment=environment,
    )


def verify_artifacts(
    artifacts: object,
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    environment: str | None = None,
) -> list[dict[str, str]]:
    """Verify a closeout's path triples resolve to existing regular files."""
    if not isinstance(artifacts, list) or len(artifacts) > 100:
        raise PathGovernanceError("local_artifacts must be a list of at most 100 path triples")
    authorized = tuple(authorized_roots)
    verified: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"WSL_PATH", "WINDOWS_UNC", "OPEN_COMMAND"}:
            raise PathGovernanceError("each local artifact must contain WSL_PATH, WINDOWS_UNC, and OPEN_COMMAND")
        if any(not isinstance(item[key], str) for key in ("WSL_PATH", "WINDOWS_UNC", "OPEN_COMMAND")):
            raise PathGovernanceError("artifact path fields must be strings")
        location = validate_output_path(
            item["WSL_PATH"],
            authorized_roots=authorized,
            require_exists=True,
            expected_kind="file",
            environment=environment,
        )
        if location.wsl_path in seen:
            raise PathGovernanceError("local_artifacts contains a duplicate path")
        seen.add(location.wsl_path)
        if item["WINDOWS_UNC"] != location.windows_unc or item["OPEN_COMMAND"] != location.open_command:
            raise PathGovernanceError("local artifact path mapping does not match the canonical path")
        try:
            mode = location.native_path.lstat().st_mode
        except OSError as exc:
            raise PathGovernanceError("advertised output cannot be inspected") from exc
        if not stat.S_ISREG(mode):
            raise PathGovernanceError("advertised output is not a regular file")
        verified.append(location.as_dict())
    return verified


def write_artifact_manifest(
    output_directory: str | os.PathLike[str],
    *,
    authorized_roots: Iterable[str | os.PathLike[str]] = (),
    environment: str | None = None,
) -> dict[str, object]:
    """Write and verify a no-overwrite manifest for every regular file in a run."""
    authorized = tuple(authorized_roots)
    root = validate_output_path(
        output_directory,
        authorized_roots=authorized,
        require_exists=True,
        expected_kind="directory",
        environment=environment,
    )
    manifest_path = root.native_path / "artifact_paths.json"
    if manifest_path.exists():
        raise PathGovernanceError("artifact manifest already exists; refusing to overwrite evidence")
    artifact_paths: list[dict[str, str]] = []
    for candidate in sorted(root.native_path.rglob("*")):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise PathGovernanceError("run output contains a symlink or junction")
        if candidate.is_file():
            artifact_paths.append(
                validate_output_path(
                    candidate,
                    authorized_roots=authorized,
                    require_exists=True,
                    expected_kind="file",
                    environment=environment,
                ).as_dict()
            )
        elif not candidate.is_dir():
            raise PathGovernanceError("run output contains a non-regular filesystem entry")
    manifest_location = validate_output_path(
        manifest_path,
        authorized_roots=authorized,
        environment=environment,
    )
    artifact_paths.append(manifest_location.as_dict())
    payload = {
        "schema_version": "robotsim.local-artifacts.v1",
        "artifacts": artifact_paths,
    }
    write_new_text(
        manifest_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        authorized_roots=authorized,
        environment=environment,
    )
    verify_artifacts(artifact_paths, authorized_roots=authorized, environment=environment)
    return {"manifest": manifest_location.as_dict(), "artifacts": artifact_paths}
