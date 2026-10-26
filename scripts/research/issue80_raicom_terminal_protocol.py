"""Atomic terminal-artifact protocol for bounded RAICOM observers.

This focused helper keeps parent cleanup from treating an early base summary as a
completed standing observer. A terminal manifest is published only after every
declared summary is durable and the JSONL trace is flushed and closed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

TERMINAL_SCHEMA = "robotsim.issue80.observer-terminal.v1"
EXIT_SCHEMA = "robotsim.issue80.observer-process-exit.v1"
TERMINAL_MANIFEST_FILENAME = "observer_terminal.json"
PROCESS_EXIT_FILENAME = "observer_process_exit.json"
BASE_SUMMARY_FILENAME = "reset_only_observer_summary.json"


class TerminalProtocolError(RuntimeError):
    """A terminal artifact is absent, incomplete, unsafe, or inconsistent."""


def _safe_child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise TerminalProtocolError(f"unsafe artifact name: {name!r}")
    path = root / name
    if path.parent != root:
        raise TerminalProtocolError(f"artifact escaped output root: {name!r}")
    return path


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TerminalProtocolError(f"artifact is missing or unsafe: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Make one new JSON artifact visible only after a durable complete write."""
    if path.is_symlink() or path.exists():
        raise TerminalProtocolError(f"refusing to overwrite terminal artifact: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise TerminalProtocolError(f"unsafe terminal artifact parent: {path.parent}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(payload, stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory replace makes the final name visible atomically. The
        # precondition and observer's one-shot completion guard reject a
        # duplicate callback before this point.
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise TerminalProtocolError(
            f"terminal artifact already exists: {path.name}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TerminalProtocolError(f"missing or unsafe JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalProtocolError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise TerminalProtocolError(f"JSON artifact is not an object: {path.name}")
    return value


def validate_complete_jsonl(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TerminalProtocolError(f"missing or unsafe trace: {path}")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise TerminalProtocolError("trace has an unterminated final record")
    records = 0
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise TerminalProtocolError(
                f"trace JSON record {line_number} is incomplete or invalid"
            ) from exc
        records += 1
    if records == 0:
        raise TerminalProtocolError("trace contains no records")
    return {"records": records, "sha256": hashlib.sha256(raw).hexdigest()}


def publish_terminal_manifest(
    output_dir: Path,
    *,
    observer_status: str,
    failure: str | None,
    summary_filenames: Sequence[str],
    trace_filename: str,
    finalization_monotonic_ns: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish terminal readiness after summaries and a closed trace are durable."""
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise TerminalProtocolError(f"unsafe output directory: {output_dir}")
    if not summary_filenames:
        raise TerminalProtocolError("terminal manifest needs at least one summary")
    summaries: dict[str, Any] = {}
    for filename in summary_filenames:
        path = _safe_child(output_dir, filename)
        summary = _load_json(path)
        summaries[filename] = {
            "sha256": sha256_file(path),
            "observer_status": summary.get("observer_status"),
            "status": summary.get("status"),
        }
    trace_path = _safe_child(output_dir, trace_filename)
    trace = validate_complete_jsonl(trace_path)
    payload: dict[str, Any] = {
        "schema": TERMINAL_SCHEMA,
        "terminal_state": "FINALIZED",
        "observer_status": observer_status,
        "failure": failure,
        "finalization_monotonic_ns": finalization_monotonic_ns,
        "summaries": summaries,
        "trace": {"filename": trace_filename, **trace},
        "all_summaries_durable_before_terminal": True,
        "trace_closed_before_terminal": True,
    }
    if extra:
        payload["extra"] = dict(extra)
    atomic_write_new_json(_safe_child(output_dir, TERMINAL_MANIFEST_FILENAME), payload)
    return payload


def validate_terminal_manifest(
    output_dir: Path,
    *,
    required_summary_filenames: Sequence[str],
) -> dict[str, Any]:
    """Verify terminal readiness without trusting merely file existence."""
    manifest_path = _safe_child(output_dir, TERMINAL_MANIFEST_FILENAME)
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != TERMINAL_SCHEMA:
        raise TerminalProtocolError("terminal manifest schema mismatch")
    if manifest.get("terminal_state") != "FINALIZED":
        raise TerminalProtocolError("terminal manifest is not finalized")
    if not manifest.get("all_summaries_durable_before_terminal"):
        raise TerminalProtocolError("terminal manifest lacks summary durability proof")
    if not manifest.get("trace_closed_before_terminal"):
        raise TerminalProtocolError("terminal manifest lacks trace-close proof")
    summaries = manifest.get("summaries")
    if not isinstance(summaries, Mapping):
        raise TerminalProtocolError("terminal summary descriptors are missing")
    for filename in required_summary_filenames:
        path = _safe_child(output_dir, filename)
        descriptor = summaries.get(filename)
        if not isinstance(descriptor, Mapping):
            raise TerminalProtocolError(f"terminal manifest omits required summary: {filename}")
        _load_json(path)
        if descriptor.get("sha256") != sha256_file(path):
            raise TerminalProtocolError(f"summary hash mismatch: {filename}")
    trace = manifest.get("trace")
    if not isinstance(trace, Mapping):
        raise TerminalProtocolError("terminal trace descriptor is missing")
    trace_path = _safe_child(output_dir, str(trace.get("filename", "")))
    validated_trace = validate_complete_jsonl(trace_path)
    if trace.get("sha256") != validated_trace["sha256"]:
        raise TerminalProtocolError("terminal trace hash mismatch")
    if trace.get("records") != validated_trace["records"]:
        raise TerminalProtocolError("terminal trace record count mismatch")
    return manifest


def publish_process_exit(
    output_dir: Path,
    *,
    returncode: int,
    timed_out: bool,
    terminal_validation: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    payload = {
        "schema": EXIT_SCHEMA,
        "process_state": "EXITED",
        "returncode": returncode,
        "timed_out": timed_out,
        "terminal_validation": dict(terminal_validation) if terminal_validation else None,
        "error": error,
        "completed_monotonic_ns": time.monotonic_ns(),
    }
    atomic_write_new_json(_safe_child(output_dir, PROCESS_EXIT_FILENAME), payload)
    return payload


def validate_process_exit(
    output_dir: Path,
    *,
    required_summary_filenames: Sequence[str],
) -> dict[str, Any]:
    exit_record = _load_json(_safe_child(output_dir, PROCESS_EXIT_FILENAME))
    if exit_record.get("schema") != EXIT_SCHEMA:
        raise TerminalProtocolError("observer exit record schema mismatch")
    if exit_record.get("process_state") != "EXITED":
        raise TerminalProtocolError("observer process did not prove exit")
    if exit_record.get("returncode") != 0 or exit_record.get("timed_out"):
        raise TerminalProtocolError(
            f"observer exit was not successful: {exit_record.get('returncode')}"
        )
    validation = exit_record.get("terminal_validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "PASS":
        raise TerminalProtocolError("observer exit lacks successful terminal validation")
    validate_terminal_manifest(
        output_dir,
        required_summary_filenames=required_summary_filenames,
    )
    return exit_record
