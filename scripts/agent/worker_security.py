"""Process isolation for the local Codex task worker."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


BASE_WORKER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
HIDDEN_ROOTS = ("/home", "/root", "/tmp", "/run", "/mnt", "/media", "/var/tmp")
WORKER_PERMISSION_PROFILE = "robotsim_worker"


def codex_permission_profile_overrides(*, read_only: bool) -> tuple[str, ...]:
    workspace_access = "read" if read_only else "write"
    filesystem = (
        '{"/" = "read", "/home" = "' + workspace_access
        + '", "/root" = "deny", "/var/tmp" = "deny", '
        '"/tmp" = "write", "/run/task-input" = "read"}'
    )
    return (
        f"default_permissions={WORKER_PERMISSION_PROFILE}",
        f"permissions.{WORKER_PERMISSION_PROFILE}.filesystem={filesystem}",
        f"permissions.{WORKER_PERMISSION_PROFILE}.network.enabled=false",
    )


@dataclass(frozen=True)
class CodexRuntime:
    host_executable: Path
    sandbox_executable: str
    mounts: tuple[str, ...]
    path_entries: tuple[str, ...]
    node_prefix: tuple[str, ...] = ()

    def map_command(self, command: Sequence[str]) -> list[str]:
        if not command or Path(command[0]).resolve() != self.host_executable:
            raise RuntimeError("Codex executable changed during worker setup")
        return [*self.node_prefix, self.sandbox_executable, *command[1:]]


def build_worker_environment(
    *,
    path_entries: Sequence[str] = (),
    ambient: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an allowlisted environment; coordinator secrets never enter it."""
    source = os.environ if ambient is None else ambient
    path = ":".join((*path_entries, BASE_WORKER_PATH))
    environment = {
        "PATH": path,
        "HOME": "/tmp/worker-home",
        "CODEX_HOME": "/root",
        "TMPDIR": "/tmp",
        "TEMP": "/tmp",
        "TMP": "/tmp",
        "XDG_CONFIG_HOME": "/tmp/worker-home/.config",
        "XDG_CACHE_HOME": "/tmp/cache",
        "XDG_STATE_HOME": "/tmp/worker-home/.local/state",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "NO_COLOR": "1",
    }
    for name in ("LANG", "LC_ALL", "TERM"):
        value = source.get(name)
        if value and "\x00" not in value and "\n" not in value:
            environment[name] = value
    return environment


def prepare_worker_codex_home(
    state_directory: Path,
    issue_number: int,
    *,
    profile: str = "writer",
    ambient: Mapping[str, str] | None = None,
) -> Path:
    """Copy only Codex auth into a private, resumable per-Issue profile."""
    source = os.environ if ambient is None else ambient
    source_home = Path(source.get("CODEX_HOME") or (Path.home() / ".codex")).expanduser()
    source_auth = source_home / "auth.json"
    if source_auth.is_symlink() or not source_auth.is_file():
        raise RuntimeError("Codex file authentication is unavailable for the isolated worker")

    if not profile or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in profile):
        raise ValueError("Codex profile name is invalid")
    root = state_directory.expanduser().resolve()
    worker_home = root / "worker-codex" / f"issue-{issue_number}-{profile}"
    if worker_home.is_symlink():
        raise RuntimeError("isolated Codex profile path is a symlink")
    worker_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(worker_home, 0o700)

    target_auth = worker_home / "auth.json"
    if target_auth.is_symlink():
        raise RuntimeError("isolated Codex auth path is a symlink")
    temporary = worker_home / f".auth-{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(source_auth, temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target_auth)
    finally:
        temporary.unlink(missing_ok=True)
    return worker_home


def _under_masked_root(path: Path) -> bool:
    normalized = path.as_posix()
    return any(normalized == root or normalized.startswith(root + "/") for root in HIDDEN_ROOTS)


def _new_runtime_target(prefix: str) -> str:
    return f"/run/robotsim-{prefix}-{uuid.uuid4().hex}"


def resolve_codex_runtime(executable: str | None = None) -> CodexRuntime:
    """Resolve a Codex launcher into a minimal read-only mount outside host HOME."""
    selected = executable or shutil.which("codex")
    if not selected:
        raise RuntimeError("Codex CLI is not installed")
    selected_path = Path(selected).expanduser()
    if selected_path.is_symlink():
        selected_path = selected_path.resolve(strict=True)
    host_executable = selected_path.resolve()
    if not host_executable.is_file():
        raise RuntimeError("Codex CLI executable is unavailable")

    mounts: list[str] = []
    path_entries: list[str] = []
    sandbox_executable = str(host_executable)
    node_prefix: tuple[str, ...] = ()
    runtime_root: Path | None = None
    relative_executable: Path | None = None
    if _under_masked_root(host_executable):
        node_modules_root = next(
            (parent for parent in host_executable.parents if parent.name == "node_modules"),
            None,
        )
        if node_modules_root is not None:
            runtime_root = node_modules_root
        else:
            runtime_root = next(
                (
                    parent for parent in host_executable.parents
                    if (parent / "node_modules" / "@openai" / "codex").is_dir()
                ),
                None,
            )
            if runtime_root is None:
                raise RuntimeError("Codex CLI runtime cannot be isolated from host credentials")
        relative_executable = host_executable.relative_to(runtime_root)
        sandbox_root = _new_runtime_target("codex")
        mounts.extend(("--dir", sandbox_root, "--ro-bind", str(runtime_root), sandbox_root))
        sandbox_executable = f"{sandbox_root}/{relative_executable.as_posix()}"

    script = host_executable.suffix.casefold() in {".js", ".cjs", ".mjs"}
    if not script:
        try:
            with host_executable.open("rb") as launcher:
                header = launcher.read(8192)
            script = header.startswith(b"#!") and b"node" in header.lower()
        except OSError:
            script = False
    if script:
        node = shutil.which("node")
        if not node:
            candidates = sorted(Path.home().glob(".nvm/versions/node/*/bin/node"))
            if len(candidates) == 1:
                node = str(candidates[0])
        if not node:
            raise RuntimeError("Node.js is required by the Codex CLI but is unavailable")
        node_path = Path(node).expanduser().resolve()
        if not node_path.is_file():
            raise RuntimeError("Node.js runtime is unavailable")
        if _under_masked_root(node_path):
            node_root = node_path.parent
            node_target = _new_runtime_target("node")
            mounts.extend(("--dir", node_target, "--ro-bind", str(node_root), node_target))
            mapped_node = f"{node_target}/{node_path.name}"
            path_entries.append(node_target)
        else:
            mapped_node = str(node_path)
            path_entries.append(str(node_path.parent))
        if host_executable.suffix.casefold() in {".js", ".cjs", ".mjs"}:
            node_prefix = (mapped_node,)
    return CodexRuntime(
        host_executable, sandbox_executable, tuple(mounts), tuple(path_entries), node_prefix,
    )


def build_isolated_command(
    command: Sequence[str],
    *,
    workspace: Path,
    codex_home: Path,
    input_directory: Path,
    output_directory: Path,
    read_only: bool = False,
    workspace_write_roots: Sequence[Path] | None = None,
    runtime: CodexRuntime | None = None,
    bubblewrap: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build a WSL/Linux namespace with only the task, Codex profile, and I/O mounts."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("isolated local Codex workers require Linux bubblewrap")
    executable = bubblewrap or shutil.which("bwrap")
    if not executable:
        raise RuntimeError("bubblewrap is required; refusing an unsandboxed worker")
    if not command or not workspace.is_dir():
        raise ValueError("worker command and workspace are required")
    runtime = runtime or resolve_codex_runtime(command[0])
    mapped_command = runtime.map_command(command)
    paths = (workspace, codex_home, input_directory, output_directory)
    for path in paths:
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError("worker isolation mount is not a real directory")
    resolved_workspace, resolved_profile, resolved_input, resolved_output = (
        path.resolve() for path in paths
    )
    resolved_write_roots: list[tuple[Path, str]] = []
    if workspace_write_roots is not None:
        for path in workspace_write_roots:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("scoped worker write root is not a real directory")
            resolved_path = path.resolve()
            try:
                relative = resolved_path.relative_to(resolved_workspace).as_posix()
            except ValueError as exc:
                raise RuntimeError("scoped worker write root escapes the task workspace") from exc
            if relative in {"", "."}:
                raise RuntimeError("scoped worker write root cannot be the whole workspace")
            resolved_write_roots.append((resolved_path, f"/home/{relative}"))

    safe_path_entries = (*runtime.path_entries,)
    environment = build_worker_environment(path_entries=safe_path_entries)
    command_line = [
        executable,
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--new-session", "--die-with-parent", "--cap-drop", "ALL",
        "--ro-bind", "/", "/",
        "--tmpfs", "/run",
        *runtime.mounts,
        "--bind", str(resolved_profile), "/root",
        "--dir", "/run/task-input", "--ro-bind", str(resolved_input), "/run/task-input",
        "--bind", str(resolved_output), "/var/tmp",
        "--bind" if not read_only and workspace_write_roots is None else "--ro-bind",
        str(resolved_workspace), "/home",
        *(
            argument
            for source, target in resolved_write_roots
            for argument in ("--bind", str(source), target)
        ),
        "--tmpfs", "/tmp", "--dir", "/tmp/worker-home", "--dir", "/tmp/cache",
        "--tmpfs", "/mnt", "--tmpfs", "/media",
        "--proc", "/proc", "--dev", "/dev", "--clearenv",
    ]
    for name, value in environment.items():
        command_line.extend(("--setenv", name, value))
    command_line.extend(("--chdir", "/home", "--", *mapped_command))
    return command_line, environment


def verify_bubblewrap() -> None:
    executable = shutil.which("bwrap")
    if not executable or not sys.platform.startswith("linux"):
        raise RuntimeError("bubblewrap isolation is unavailable; refusing an unsandboxed worker")
    result = subprocess.run(
        [executable, "--unshare-user", "--unshare-pid", "--ro-bind", "/", "/",
         "--proc", "/proc", "--dev", "/dev", "--", "/usr/bin/true"],
        env=build_worker_environment(),
        cwd="/",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError("bubblewrap user, process, and filesystem namespaces are unavailable")
