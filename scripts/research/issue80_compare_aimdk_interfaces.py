#!/usr/bin/env python3
"""Read-only comparison of an AimDK SDK archive and installed ROS interfaces."""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ARCHIVE_ROOT_PATTERN = re.compile(r"(?P<root>.+/src/aimdk_msgs/)package\.xml$")
ARTIFACT_PATTERN = re.compile(r"aimdk-(?P<arch>[a-zA-Z0-9_]+)-(?P<revision>[0-9a-f]+)-artifacts")


def _source_lines(text: str, tokens: tuple[str, ...]) -> list[str]:
    return [line.strip() for line in text.splitlines() if any(token in line for token in tokens)]


def compare_interfaces(
    archive_path: Path,
    runtime_package: Path,
    runtime_set_mode_example: Path | None = None,
) -> dict[str, Any]:
    """Compare package.xml and every .msg/.srv source file without extraction."""
    archive_path = archive_path.resolve(strict=True)
    runtime_package = runtime_package.resolve(strict=True)
    if not runtime_package.is_dir():
        raise NotADirectoryError(runtime_package)

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        package_members = sorted(
            name for name in names if ARCHIVE_ROOT_PATTERN.fullmatch(name)
        )
        if len(package_members) != 1:
            raise ValueError(f"expected exactly one SDK aimdk_msgs package.xml, found {len(package_members)}")
        package_member = package_members[0]
        prefix = package_member[: -len("package.xml")]
        sdk_package_xml = archive.read(package_member)
        runtime_package_xml = (runtime_package / "package.xml").read_bytes()
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        mismatched: list[str] = []

        for member in sorted(
            name for name in names
            if name.startswith(prefix + "interface/") and name.endswith((".msg", ".srv"))
        ):
            relative = member[len(prefix):]
            sdk_bytes = archive.read(member)
            runtime_file = runtime_package / relative
            runtime_bytes = runtime_file.read_bytes() if runtime_file.is_file() else None
            identical = runtime_bytes == sdk_bytes
            rows.append({
                "path": relative,
                "byte_identical": identical,
                "sdk_sha256": sha256(sdk_bytes).hexdigest(),
                "runtime_sha256": sha256(runtime_bytes).hexdigest() if runtime_bytes is not None else None,
            })
            if runtime_bytes is None:
                missing.append(relative)
            elif not identical:
                mismatched.append(relative)

        examples: dict[str, Any] = {}
        for member in sorted(names):
            if member.endswith("/set_mc_action.py") and "src/py_examples/" in member:
                examples["set_mc_action"] = {
                    "path": member,
                    "source": _source_lines(archive.read(member).decode("utf-8", errors="replace"), ("req.source",)),
                    "action_desc": _source_lines(archive.read(member).decode("utf-8", errors="replace"), ("cmd.action_desc",)),
                }
            elif member.endswith("/set_mc_input_source.py") and "src/py_examples/" in member:
                text = archive.read(member).decode("utf-8", errors="replace")
                examples["set_mc_input_source"] = {
                    "path": member,
                    "fields": _source_lines(text, ("req.action.value", "req.input_source.name", "req.input_source.priority", "req.input_source.timeout")),
                }

    match = ARTIFACT_PATTERN.search(package_member)
    runtime_source_lines = []
    if runtime_set_mode_example is not None:
        runtime_source_lines = _source_lines(
            runtime_set_mode_example.read_text(encoding="utf-8"), ("source =", "req.source =")
        )
    return {
        "schema": "robotsim.issue80.aimdk-interface-comparison.v1",
        "sdk_archive_name": archive_path.name,
        "sdk_archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "sdk_package_root": prefix,
        "sdk_artifact_architecture": match.group("arch") if match else None,
        "sdk_artifact_revision": match.group("revision") if match else None,
        "runtime_package_root": str(runtime_package),
        "sdk_package_xml_sha256": sha256(sdk_package_xml).hexdigest(),
        "runtime_package_xml_sha256": sha256(runtime_package_xml).hexdigest(),
        "package_xml_byte_identical": sdk_package_xml == runtime_package_xml,
        "interface_file_count": len(rows),
        "interface_byte_identical_count": sum(row["byte_identical"] for row in rows),
        "missing_runtime_files": missing,
        "mismatched_runtime_files": mismatched,
        "sdk_examples": examples,
        "runtime_set_mode_example_source_lines": runtime_source_lines,
        "interfaces": rows,
        "all_source_interfaces_match": (
            bool(rows) and not missing and not mismatched and sdk_package_xml == runtime_package_xml
        ),
        "scope_note": (
            "Source interface parity only; the archive has architecture-specific AArch64 prebuilts and the runtime uses x86_64. "
            "This does not prove binary ABI compatibility or MC implementation behavior."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runtime-package", type=Path, required=True, help="installed source package root containing package.xml and interface/")
    parser.add_argument("--runtime-set-mode-example", type=Path)
    parser.add_argument("--output", type=Path, help="optional new output file inside a reserved evidence directory")
    args = parser.parse_args()
    result = compare_interfaces(args.archive, args.runtime_package, args.runtime_set_mode_example)
    output_text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        repository_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repository_root / "scripts" / "agent"))
        from path_governance import write_new_text
        write_new_text(args.output, output_text)
    else:
        print(output_text, end="")
    return 0 if result["all_source_interfaces_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
