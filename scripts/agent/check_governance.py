#!/usr/bin/env python3
"""Check direct runtime/research boundaries and obvious tracked run artifacts."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOTS = ("apps/", "robots/", "ros2_ws/src/", "simulation/")
SOURCE_SUFFIXES = {
    ".bat", ".bash", ".c", ".cc", ".cmd", ".cpp", ".cs", ".cxx",
    ".h", ".hh", ".hpp", ".hxx", ".json", ".toml", ".ps1", ".py",
    ".sh", ".txt", ".xml", ".yaml", ".yml", ".cmake",
}
GENERATED_DIRECTORIES = {
    "artifacts", "captures", "evidence", "frames", "generated", "logs",
    "outputs", "recordings", "rendered", "reviews", "screenshots", "scratch",
    "temp", "tmp", "traces",
}
GENERATED_SUFFIXES = {".jsonl", ".log", ".mp4"}
LARGE_DATA_SUFFIXES = {".csv", ".json"}
SMALL_FIXTURE_MAX_BYTES = 1_048_576
FIXTURE_ROOTS = ("data/fixtures/", "tests/fixtures/")
REQUIRED_POLICY_DOCUMENTS = {
    "docs/engineering/runtime.md",
    "docs/engineering/validation.md",
}
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
FROM_SCRIPTS_IMPORT_RESEARCH = re.compile(
    r"(?m)^\s*from\s+scripts\s+import\s+[^#\n]*\bresearch\b"
)
RESEARCH_PATH_MARKERS = ("scripts/research", "scripts.research")


def _tracked_paths(root: Path) -> tuple[list[str], str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"could not enumerate tracked files ({type(exc).__name__})"
    if result.returncode != 0:
        return [], "could not enumerate tracked files"
    return [os.fsdecode(item) for item in result.stdout.split(b"\0") if item], None


def _policy_reference_problems(root: Path) -> list[str]:
    agents_path = root / "AGENTS.md"
    try:
        text = agents_path.read_text(encoding="utf-8")
    except OSError:
        return ["AGENTS.md is required to reference the runtime and validation contracts"]

    referenced: set[str] = set()
    for target in MARKDOWN_LINK.findall(text):
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        relative = target.split("#", 1)[0].split("?", 1)[0]
        if relative:
            referenced.add((PurePosixPath("AGENTS.md").parent / relative).as_posix())

    problems: list[str] = []
    for document in sorted(REQUIRED_POLICY_DOCUMENTS):
        if document not in referenced:
            problems.append(f"AGENTS.md must link to {document}")
        if not (root / document).is_file():
            problems.append(f"required policy document is missing: {document}")
    return problems


def _is_small_fixture(path: str, size: int, is_symlink: bool) -> bool:
    return (
        not is_symlink
        and size <= SMALL_FIXTURE_MAX_BYTES
        and any(path.startswith(prefix) for prefix in FIXTURE_ROOTS)
    )


def _artifact_problem(path: str, root: Path) -> str | None:
    source_path = PurePosixPath(path)
    suffix = source_path.suffix.casefold()
    full_path = root / path
    try:
        metadata = full_path.lstat()
        size = metadata.st_size
        is_symlink = full_path.is_symlink()
    except OSError:
        size = 0
        is_symlink = False

    if _is_small_fixture(path, size, is_symlink):
        return None

    generated_directory = next(
        (part for part in source_path.parts[:-1] if part.casefold() in GENERATED_DIRECTORIES),
        None,
    )
    if generated_directory is not None:
        return f"tracked under generated-output directory '{generated_directory}'"
    if suffix in GENERATED_SUFFIXES:
        return f"tracked generated-output file type '{suffix}'"
    if suffix in LARGE_DATA_SUFFIXES and size > SMALL_FIXTURE_MAX_BYTES:
        return f"tracked data file exceeds {SMALL_FIXTURE_MAX_BYTES} bytes"
    return None


def _has_research_dependency(path: str, root: Path) -> bool:
    source_path = root / path
    try:
        text = os.readlink(source_path) if source_path.is_symlink() else source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False

    normalized = text.casefold().replace("\\", "/")
    if any(marker in normalized for marker in RESEARCH_PATH_MARKERS):
        return True
    if FROM_SCRIPTS_IMPORT_RESEARCH.search(text):
        return True
    if source_path.suffix.casefold() != ".py":
        return False

    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "scripts.research" or alias.name.startswith("scripts.research.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "scripts.research" or module.startswith("scripts.research."):
                return True
            if module == "scripts" and any(alias.name == "research" for alias in node.names):
                return True
    return False


def check_repository(root: str | os.PathLike[str]) -> list[str]:
    """Return findings for policy links, direct runtime dependencies, and tracked outputs."""
    repository = Path(root).resolve()
    tracked, problem = _tracked_paths(repository)
    if problem:
        return [problem]

    problems = _policy_reference_problems(repository)
    for path in tracked:
        artifact_problem = _artifact_problem(path, repository)
        if artifact_problem:
            problems.append(f"{path}: {artifact_problem}")
        if path.startswith(RUNTIME_ROOTS) and PurePosixPath(path).suffix.casefold() in SOURCE_SUFFIXES:
            if _has_research_dependency(path, repository):
                problems.append(f"{path}: runtime source directly references scripts/research")
    return problems


def main() -> int:
    problems = check_repository(REPOSITORY_ROOT)
    if problems:
        for problem in problems:
            print(f"governance check: {problem}", file=sys.stderr)
        return 1
    print("Runtime/research boundary, tracked-output, and policy-link checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
