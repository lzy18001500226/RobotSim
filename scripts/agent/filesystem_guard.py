"""Conservative checks for visible shell deletion/disk commands, not a sandbox."""

from __future__ import annotations

import re
import shlex
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
PROTECTED = {".git", "reference", "third_party", "reviews"}
SHELLS = {"sh", "bash", "zsh", "dash", "cmd", "powershell", "pwsh"}
DISK_COMMANDS = {"format", "diskpart", "wipefs", "blkdiscard", "fdisk", "sfdisk", "parted"}
DELETE_COMMANDS = {"rm", "rmdir", "rd", "remove-item", "del", "erase", "ri"}


def _literal(token: str) -> str:
    return token.strip("\"'")


def _executable(token: str) -> str:
    return _literal(token).replace("\\", "/").rsplit("/", 1)[-1].casefold().removesuffix(".exe").removesuffix(".com")


def _tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=False, punctuation_chars=";&|<>\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    return list(lexer)


def _scoped_path(token: str, cwd: str | None) -> Path | None:
    value = _literal(token)
    # Opaque expressions and glob expansion cannot establish a literal scope.
    if not value or re.search(r"[$`*?{}\[\]\n\r]", value) or value.startswith("~"):
        return None
    normalized = value.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", normalized) or normalized.startswith("//"):
        return None
    path = Path(value)
    if not path.is_absolute():
        if not cwd or not Path(cwd).is_absolute():
            return None
        path = Path(cwd) / path
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(REPOSITORY.resolve())
    except (ValueError, OSError, RuntimeError):
        return None
    if not relative.parts or any(part in PROTECTED for part in relative.parts):
        return None
    return resolved


def _delete_reason(args: list[str], cwd: str | None) -> str | None:
    targets = [arg for arg in args if not _literal(arg).startswith("-") and _literal(arg).casefold() not in {"/s", "/q"}]
    if not targets or any(_scoped_path(target, cwd) is None for target in targets):
        return "blocked deletion without literal, non-protected repository targets; review resolved paths first"
    return None


def command_reason(command: str, cwd: str | None = None, *, depth: int = 0) -> str | None:
    """Inspect direct commands and common shell wrappers; never execute input."""
    if depth > 4:
        return "blocked deeply nested shell command that cannot be inspected safely"
    try:
        tokens = _tokens(command)
    except ValueError:
        return "blocked malformed shell command that cannot be inspected safely"
    # Segment commands so quoted code/data does not become an executable name.
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in {";", "&&", "||", "|", "&"} or (token and not token.strip("\n")):
            segments.append([])
        else:
            segments[-1].append(token)
    for segment in segments:
        for index, token in enumerate(segment):
            # Only command positions and transparent launch wrappers are inspected.
            if index and _executable(segment[0]) not in {"sudo", "env", "command", "wsl"}:
                break
            name = _executable(token)
            args = segment[index + 1:]
            if name in DISK_COMMANDS or name == "mkfs" or name.startswith("mkfs."):
                return "blocked a disk formatting/partitioning command; use a separately reviewed manual operation"
            if name == "dd" and any(re.match(r"of=/dev/", _literal(arg)) for arg in args):
                return "blocked raw disk output through dd"
            if name in SHELLS:
                for offset, arg in enumerate(args[:-1]):
                    flag = _literal(arg).casefold()
                    if flag in {"-c", "-command", "/c"} or (name in {"sh", "bash", "zsh", "dash"} and re.fullmatch(r"-[a-z]*c", flag)):
                        reason = command_reason(_literal(args[offset + 1]), cwd, depth=depth + 1)
                        if reason:
                            return reason
            if name in DELETE_COMMANDS:
                reason = _delete_reason(args, cwd)
                if reason:
                    return reason
            # A preceding cd changes interpretation; relative deletes then fail closed.
            if name in {"cd", "set-location", "pushd"}:
                cwd = None
    return None
