#!/usr/bin/env python3
"""Extract one source-local, SD-stable MuJoCo qpos posture for offline LiDAR maps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


class PostureError(ValueError):
    pass


def extract(trace: Path) -> dict:
    if trace.is_symlink() or not trace.is_file():
        raise PostureError("native MjData trace is missing or unsafe")
    metadata = None
    samples: list[dict] = []
    raw_lines = trace.read_bytes().splitlines(keepends=True)
    for index, raw in enumerate(raw_lines, start=1):
        if not raw.endswith(b"\n"):
            if index == len(raw_lines):
                break
            raise PostureError("native MjData trace contains an incomplete non-final record")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PostureError(f"invalid native MjData JSON at line {index}") from exc
        if row.get("event") == "model_metadata":
            metadata = row
        elif row.get("event") == "post_step" and isinstance(row.get("qpos"), list):
            samples.append(row)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("nq"), int):
        raise PostureError("native MjData trace lacks model nq metadata")
    nq = int(metadata["nq"])
    if not samples:
        raise PostureError("native MjData trace has no complete post-step qpos sample")
    generation = max(int(row.get("generation", -1)) for row in samples)
    current = [row for row in samples if int(row.get("generation", -1)) == generation]
    sample = current[-1]
    qpos = sample["qpos"]
    if len(qpos) != nq:
        raise PostureError("selected qpos does not match model nq")
    try:
        values = [float(item) for item in qpos]
    except (TypeError, ValueError) as exc:
        raise PostureError("qpos has a nonnumeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise PostureError("qpos has a nonfinite value")
    pelvis = sample.get("pelvis", {}).get("state", {})
    if pelvis.get("name") != "pelvis":
        raise PostureError("selected trace sample lacks named pelvis state")
    return {
        "schema": "robotsim.issue80.lidar-standing-posture.v1",
        "classification": "SIMULATION_ONLY_READ_ONLY_SD_POSTURE_EXTRACTION",
        "status": "PASS",
        "source_trace": str(trace),
        "source_generation": generation,
        "source_sim_time_s": sample.get("sim_time_s"),
        "source_host_monotonic_ns": sample.get("host_monotonic_ns"),
        "nq": nq,
        "qpos": values,
        "pelvis": pelvis,
        "mj_step_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--qpos-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.result.exists() or args.qpos_file.exists():
        parser.error("result and qpos-file must be new")
    try:
        result = extract(args.trace)
    except (OSError, PostureError) as exc:
        parser.error(str(exc))
    args.qpos_file.write_text(" ".join(format(value, ".17g") for value in result["qpos"]) + "\n", encoding="utf-8")
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
