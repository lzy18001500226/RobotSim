"""Create the tracked static signature fixture from the frozen PR #90 MJCF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco

from .model_signature import compiled_model_signature, signature_sha256


FROZEN_SHA256 = "32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-mjcf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.reference_mjcf.is_file():
        raise SystemExit(f"missing frozen reference MJCF: {args.reference_mjcf}")
    source_sha = hashlib.sha256(args.reference_mjcf.read_bytes()).hexdigest()
    if source_sha != FROZEN_SHA256:
        raise SystemExit("reference MJCF SHA-256 does not match the frozen PR #90 model")
    model = mujoco.MjModel.from_xml_path(str(args.reference_mjcf))
    signature = compiled_model_signature(model, args.reference_mjcf)
    payload = {
        "schema_version": "robotsim.x2-omnipicker-compiled-signature.v1",
        "reference_model_sha256": source_sha,
        "mujoco_version": mujoco.__version__,
        "signature_sha256": signature_sha256(signature),
        "signature": signature,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "SIGNATURE_CAPTURED", "sha256": payload["signature_sha256"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
