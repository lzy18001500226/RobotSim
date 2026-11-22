"""Export an explicit source scope for a stage handoff; never execute or publish it."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from scripts.agent import path_governance as paths
from scripts.agent.pre_tool_guard import _is_sensitive_path


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "--no-optional-locks", "--literal-pathspecs", "-C", str(repo), *args],
                            capture_output=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError("Git inspection failed: " + args[0])
    return result.stdout


def _sources(repo: Path, names: list[str], deleted: list[str] | None = None) -> tuple[dict, dict]:
    deleted = deleted or []
    if not names and not deleted:
        raise ValueError("an explicit nonempty source scope is required")
    if set(names) & set(deleted):
        raise ValueError("a source cannot also be a declared deletion")
    metadata, content = {}, {}
    for name in sorted(set(names) | set(deleted)):
        relative = PurePosixPath(name)
        if (not relative.parts or relative.is_absolute() or "\\" in name or relative.as_posix() != name
                or any(part in {"..", ".git", "reference", "third_party", "reviews"} for part in relative.parts)
                or (relative.parts[0] == ".codex" and name != ".codex/hooks.json")
                or _is_sensitive_path(name)):
            raise ValueError("source path is outside the exportable scope")
        source = repo / name
        if any((repo / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
            raise ValueError("symlink source is not a recoverable regular-file snapshot")
        source.resolve().relative_to(repo)
        if name in deleted:
            if source.exists():
                raise ValueError("declared deletion still exists")
            metadata[name] = {"deleted": True}
            continue
        if not source.is_file():
            raise ValueError("source must be an existing regular file")
        data = source.read_bytes()
        metadata[name] = {"sha256": hashlib.sha256(data).hexdigest(), "mode": source.stat().st_mode & 0o777}
        content[name] = data
    return metadata, content


def fingerprint(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(repo: Path, *, stage: str, names: list[str], evidence: list[Path],
           reproduction: str, validated_fingerprint: str, output_parent: Path,
           verify_published: bool = False, authorized_roots: tuple = (),
           deleted: list[str] | None = None) -> Path:
    repo = repo.resolve()
    if Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
        raise ValueError("repo must be the explicit Git checkout root")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", stage) or not reproduction.strip() or not evidence:
        raise ValueError("stage ID, reproduction command and evidence are required")
    metadata, content = _sources(repo, names, deleted)
    if fingerprint(metadata) != validated_fingerprint:
        raise ValueError("current source differs from the stage's validated fingerprint")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    branch = _git(repo, "branch", "--show-current").decode().strip()
    scope = sorted(metadata)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--", *scope).decode()
    publication = "NOT_CHECKED"
    if verify_published:
        if not branch or branch in {"main", "master"} or status:
            raise ValueError("published checkpoint needs a clean explicit scope on a task branch")
        tree = {}
        for record in _git(repo, "ls-tree", "--full-tree", "-z", "HEAD", "--", *scope).split(b"\0"):
            if record:
                info, name = record.split(b"\t", 1)
                tree[name.decode()] = info.decode().split()
        for name, item in metadata.items():
            entry = tree.get(name)
            if item.get("deleted"):
                if entry:
                    raise ValueError("declared deletion is not committed")
            elif (not entry or entry[1] != "blob" or entry[0] not in {"100644", "100755"}
                  or hashlib.sha256(_git(repo, "cat-file", "blob", entry[2])).hexdigest() != item["sha256"]
                  or (entry[0] == "100755") != bool(item["mode"] & 0o111)):
                raise ValueError("selected source content or executable mode is not committed")
        remote = _git(repo, "ls-remote", "--heads", "origin", "refs/heads/" + branch).decode().split()
        if remote != [head, "refs/heads/" + branch]:
            raise ValueError("task branch is not pushed at the inspected HEAD")
        publication = "VERIFIED"
    evidence_records = []
    for item in evidence:
        location = paths.validate_output_path(item, authorized_roots=authorized_roots,
                                             require_exists=True, expected_kind="file")
        if not location.native_path.stat().st_size:
            raise ValueError("stage evidence must be nonempty")
        evidence_records.append({"path": str(location.native_path), "sha256": _hash_file(location.native_path)})
    patch = _git(repo, "diff", "--binary", "HEAD", "--", *scope)
    run = paths.create_unique_output_directory(output_parent, prefix="stage-" + stage,
                                              authorized_roots=authorized_roots).native_path
    for name, data in content.items():
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        paths.write_new_bytes(destination, data, authorized_roots=authorized_roots)
        destination.chmod(metadata[name]["mode"])
        if _hash_file(destination) != metadata[name]["sha256"]:
            raise ValueError("source copy failed integrity verification")
    # A producer changing source, HEAD, index or evidence during export cannot
    # obtain a completed packet. Leave any partial directory for inspection.
    after, _ = _sources(repo, names, deleted)
    if (after != metadata or _git(repo, "rev-parse", "HEAD").decode().strip() != head
            or _git(repo, "branch", "--show-current").decode().strip() != branch
            or _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--", *scope).decode() != status
            or any(_hash_file(Path(item["path"])) != item["sha256"] for item in evidence_records)):
        raise ValueError("stage inputs changed during export; no completed handoff written")
    paths.write_new_bytes(run / "review.patch", patch, authorized_roots=authorized_roots)
    packet = {"schema_version": "robotsim.stage-handoff.v1", "stage_id": stage,
              "state": "HANDOFF_READY", "branch": branch, "head_sha": head,
              "source_fingerprint": validated_fingerprint, "sources": metadata,
              "scoped_git_status": status, "publication": publication,
              "reproduction_command": reproduction, "evidence": evidence_records,
              "acceptance": "producer assertion; exporter checks source identity and file integrity only",
              "task_completed": False}
    paths.write_new_text(run / "stage_handoff.json", json.dumps(packet, indent=2) + "\n",
                         authorized_roots=authorized_roots)
    paths.write_artifact_manifest(run, authorized_roots=authorized_roots)
    return run / "stage_handoff.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--deleted-source", action="append", default=[])
    parser.add_argument("--fingerprint-only", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--evidence", action="append", type=Path)
    parser.add_argument("--reproduce")
    parser.add_argument("--validated-fingerprint")
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--verify-published", action="store_true")
    args = parser.parse_args()
    try:
        if args.fingerprint_only:
            print(fingerprint(_sources(args.repo.resolve(), args.source, args.deleted_source)[0]))
        else:
            if not all((args.stage, args.evidence, args.reproduce, args.validated_fingerprint, args.output_parent)):
                parser.error("export requires stage, evidence, reproduce, validated-fingerprint and output-parent")
            print(export(args.repo, stage=args.stage, names=args.source, evidence=args.evidence,
                         reproduction=args.reproduce, validated_fingerprint=args.validated_fingerprint,
                         output_parent=args.output_parent, verify_published=args.verify_published,
                         deleted=args.deleted_source))
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print("Stage handoff unavailable: " + str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
