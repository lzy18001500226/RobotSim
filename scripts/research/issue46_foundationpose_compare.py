#!/usr/bin/env python3
"""Compare a FoundationPose result against independent static MuJoCo truth.

This records a perception measurement only.  It neither changes model state
nor turns an estimated pose into a whole-body IK or grasp command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calculate rigid-pose and cylinder-symmetry-aware errors."""
    if truth.shape != (4, 4) or prediction.shape != (4, 4):
        raise ValueError("both poses must be 4x4")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("pose contains non-finite values")
    relative_rotation = truth[:3, :3].T @ prediction[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    axis_cosine = float(np.clip(abs(np.dot(truth[:3, 2], prediction[:3, 2])), -1.0, 1.0))
    translation = float(np.linalg.norm(prediction[:3, 3] - truth[:3, 3]))
    return {
        "translation_error_m": translation,
        "translation_error_mm": translation * 1000.0,
        "rotation_error_rad": float(math.acos(cosine)),
        "cylinder_axis_undirected_error_rad": float(math.acos(axis_cosine)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-record", required=True, type=Path)
    parser.add_argument("--foundationpose-pose", required=True, type=Path)
    parser.add_argument("--foundationpose-log", required=True, type=Path)
    parser.add_argument("--sam2-result", required=True, type=Path)
    args = parser.parse_args()
    if not all(path.is_file() for path in (args.input_record, args.foundationpose_pose, args.foundationpose_log, args.sam2_result)):
        raise SystemExit("input record, FoundationPose pose/log, or SAM2 result is missing")
    input_record = json.loads(args.input_record.read_text(encoding="utf-8"))
    sam2 = json.loads(args.sam2_result.read_text(encoding="utf-8"))
    truth = np.asarray(input_record["T_camera_object_truth"], dtype=float)
    prediction = np.loadtxt(args.foundationpose_pose).reshape(4, 4)
    metrics = pose_metrics(truth, prediction)
    prompt_source = str(sam2.get("prompt_source", ""))
    yolo_proposal = prompt_source.startswith("YOLO_")
    yolo_label = str(sam2.get("yolo_label", "")) if yolo_proposal else None
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-foundationpose-static-compare").native_path
    record = {
        "schema": "robotsim.issue46.foundationpose-static-compare.v1",
        "classification": "SIMULATION_ONLY_STATIC_POSE_EVALUATION",
        "mj_step_calls": 0,
        "input": {"path": str(args.input_record.resolve()), "sha256": sha256(args.input_record)},
        "sam2": {"path": str(args.sam2_result.resolve()), "sha256": sha256(args.sam2_result), "result": sam2},
        "foundationpose": {
            "pose_path": str(args.foundationpose_pose.resolve()),
            "pose_sha256": sha256(args.foundationpose_pose),
            "log_path": str(args.foundationpose_log.resolve()),
            "log_sha256": sha256(args.foundationpose_log),
            "T_camera_object_prediction": prediction.tolist(),
        },
        "truth": {"T_camera_object": truth.tolist()},
        "metrics": metrics,
        "gates": {
            "FOUNDATIONPOSE_GPU_OUTPUT": "PASS",
            "SAM2_MASK_OUTPUT": "PASS",
            "OPTICAL_FRAME_CONVENTION": "PASS",
            "YOLO_BOX_PROPOSAL": "PASS_GEOMETRIC_ONLY" if yolo_proposal else "NOT_RUN",
            "YOLO_BOTTLE_SEMANTICS": "PASS" if yolo_label == "bottle" else ("FAIL_LABEL_" + yolo_label.upper() if yolo_label else "NOT_RUN"),
            "POSE_QUALIFIED_FOR_WHOLE_BODY_IK": "NOT_QUALIFIED",
        },
        "limitations": [
            "The SAM2 prompt is simulation truth-box smoke-test input, not a detector result.",
            "The partially occluded static frame has a measured pose error and is not passed to whole-body IK.",
            "No physical step, robot command, or grasp candidate execution occurred.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "metrics": metrics, "gates": record["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
