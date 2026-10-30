#!/usr/bin/env python3
"""Prove whether the delivered x86 RAICOM runtime actually contains X2 PNC.

The AimDK message package intentionally ships ``PncTaskRequest`` even when
the optional task-engine/planning-navigation component is absent.  This audit
keeps that crucial distinction explicit without starting Docker, MuJoCo, MC,
ROS, or a task request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


RUNTIME = Path("/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026")
ARCHIVE_ROOT = Path("/mnt/c/Users/HP/Desktop/RobotSim/AgiBot/Lingxi-X2")
COMMUNITY = Path("/home/lzy18001500226/projects/RobotSim/reference/AgiBot-Community/raicom2026-agibot-X2")
MC_ARCHIVE = ARCHIVE_ROOT / "mc-x86-v1.0.0-20260522.zip"
SIM_ARCHIVE = ARCHIVE_ROOT / "sim_mujoco-x86-v1.0.0-20260522.zip"
EXPECTED_ARCHIVE_SHA256 = {
    MC_ARCHIVE.name: "b3bc2417aad656b6a7f19b2274f07d5cb8c7f60b1317482e0122244e57d812d0",
    SIM_ARCHIVE.name: "fab28561b5c55767d3775f94c1bb94bad840a3f7c8215e22dccff5ab84a9dc5e",
}
PNC_TERMS = ("pnc", "task_engine", "task-master", "task_master", "agivslam", "slam")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_tree_matches(root: Path) -> list[str]:
    matches: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        lower = path.name.lower()
        if any(term in lower for term in PNC_TERMS):
            matches.append(str(path.relative_to(root)))
    return sorted(matches)


def archive_matches(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(
            name for name in archive.namelist()
            if any(term in name.lower() for term in PNC_TERMS)
        )


def audit() -> dict:
    required = (RUNTIME / "mc", RUNTIME / "sim_mujoco", COMMUNITY, MC_ARCHIVE, SIM_ARCHIVE)
    if any(path.is_symlink() or not path.exists() for path in required):
        raise ValueError("runtime/community/archive input is missing or unsafe")
    message = (
        RUNTIME
        / "aimdk_msgs_ws/src/aimdk_msgs/prebuilt_x86_64/local/lib/python3.10/dist-packages/aimdk_msgs/msg/_pnc_task_request.py"
    )
    if message.is_symlink() or not message.is_file():
        raise ValueError("x86 AimDK message source lacks PncTaskRequest")
    archive_hashes = {path.name: sha256(path) for path in (MC_ARCHIVE, SIM_ARCHIVE)}
    if archive_hashes != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("original x86 archive hashes differ from the pinned delivery")
    runtime_matches = {
        "mc": regular_tree_matches(RUNTIME / "mc"),
        "sim_mujoco": regular_tree_matches(RUNTIME / "sim_mujoco"),
    }
    archive_contents = {
        MC_ARCHIVE.name: archive_matches(MC_ARCHIVE),
        SIM_ARCHIVE.name: archive_matches(SIM_ARCHIVE),
    }
    # The competition repository is a physical ARM64 application: it can
    # publish the documented PNC request but does not supply x86 PNC binaries.
    activation = COMMUNITY / "scripts/activate_robot_runtime.sh"
    readme = COMMUNITY / "README.md"
    if any(path.is_symlink() or not path.is_file() for path in (activation, readme)):
        raise ValueError("community reference checkout is incomplete")
    activation_text = activation.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    community_contract = {
        "commit": _git_head(COMMUNITY),
        "activation_requires_aarch64_runtime": "aarch64_py310" in activation_text,
        "readme_declares_arm64_hardware": "ARM64" in readme_text or "arm64" in readme_text,
        "pnc_client_source": "src/race_task/race_task/task1_auto.py",
        "pnc_client_exists": (COMMUNITY / "src/race_task/race_task/task1_auto.py").is_file(),
    }
    absent = not runtime_matches["mc"] and not runtime_matches["sim_mujoco"] and not archive_contents[MC_ARCHIVE.name] and not archive_contents[SIM_ARCHIVE.name]
    return {
        "schema": "robotsim.issue80.pnc-runtime-audit.v1",
        "classification": "READ_ONLY_OFFICIAL_PNC_DELIVERY_AUDIT",
        "status": "BLOCKED_PNC_MODULE_ABSENT_FROM_X86_SIM_DELIVERY" if absent else "UNEXPECTED_PNC_ARTIFACT_FOUND",
        "installed_runtime": {
            "root": str(RUNTIME),
            "pnc_message_source": str(message),
            "pnc_message_sha256": sha256(message),
            "runtime_component_filename_matches": runtime_matches,
            "interpretation": "PncTaskRequest message support is transport/schema support only; it is not evidence that task_engine or pnc is installed.",
        },
        "original_archives": {
            "sha256": archive_hashes,
            "pnc_or_task_engine_member_matches": archive_contents,
            "read_only_inputs": True,
        },
        "community_reference": community_contract,
        "official_sources": {
            "x2_navigation_optional": "https://x2-aimdk.agibot.com/en/latest/Interface/perception/navigation.html",
            "x2_locomotion": "https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/locomotion.html",
            "software_module_topology": "https://open.agibot.com/docs/en/aimdk/a2/v2_1/dev_guide/02-software_overview",
        },
        "conclusion": {
            "pnc_client_can_be_written": True,
            "pnc_runtime_can_be_started_from_this_x86_delivery": not absent,
            "supported_next_dependency": (
                "A version-matched x86 task_engine/pnc/slam/map package and its startup configuration, "
                "or a simulator release that includes those optional components."
            ),
            "unsupported_substitutes": [
                "Treating the message definition as a running navigation server",
                "Running the ARM64 physical-robot community runtime inside the x86 simulator",
                "Disabling source joint-limit protection to force low-level turning",
            ],
        },
        "mj_step_calls": 0,
        "docker_or_ros_started": False,
        "velocity_publisher_created": False,
    }


def _git_head(path: Path) -> str:
    head = path / ".git/HEAD"
    if not head.is_file():
        return "UNKNOWN"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = path / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else "PACKED_REF_UNRESOLVED"
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve(strict=True)
    if output.is_symlink() or any(item.name != ".robotsim-output-reservation.json" for item in output.iterdir()):
        parser.error("output directory must contain only its governed reservation")
    try:
        result = audit()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(f"PNC runtime audit failed: {type(exc).__name__}: {exc}")
    (output / "pnc_runtime_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
