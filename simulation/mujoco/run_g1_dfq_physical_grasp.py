#!/usr/bin/env python3
"""Run the pinned G1 + Inspire DFQ physical grasp sequence."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "simulation/mujoco/fixtures/g1_dfq_m0_c2"
EVIDENCE = Path(os.environ.get(
    "ROBOTSIM_EVIDENCE_ROOT",
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-physical-grasp-priority-p0-20261009",
))
IMPLEMENTATION = Path(__file__).resolve().parent
C2_RUN = FIXTURE
CANDIDATE_PATH = C2_RUN / "candidate_c2.json"
WRENCH_PATH = C2_RUN / "candidate_c2_robust_wrench.json"
BASE_PATH = FIXTURE / "canonical_ready_wrench_input.json"
sys.path.insert(0, str(IMPLEMENTATION))

import g1_dfq_grasp_core as runner  # noqa: E402
import g1_inspire_hand as hand  # noqa: E402
from g1_dfq_grasp_gates import GATE_NAMES, evaluate_gates  # noqa: E402

runner.REPO = PROJECT_ROOT
runner.SCENE = FIXTURE / "g1_inspire_dfq_accepted_scene.xml"
runner.UNITREE_ROS = Path(os.environ.get(
    "ROBOTSIM_UNITREE_ROS_DIR", "/tmp/robotsim-issue43-g1-physical-grasp/upstream/unitree_ros"
))
runner.UNITREE_MUJOCO = Path(os.environ.get(
    "ROBOTSIM_UNITREE_MUJOCO_DIR", "/tmp/robotsim-issue43-stock-arm-smoke/upstream/unitree_mujoco"
))
runner.UPSTREAM = Path(os.environ.get(
    "ROBOTSIM_HUMANOID_VLA_DIR", "/tmp/robotsim-g1-dfq/upstream/humanoid_vla"
))
runner.URDF = runner.UNITREE_ROS / "robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
hand.URDF = runner.URDF


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def set_joint(model, data, name: str, value: float) -> None:
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if jid < 0:
        raise RuntimeError(f"missing model joint {name}")
    data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def apply_mimics(model, data, mimics: dict) -> None:
    for _ in range(len(mimics) + 1):
        for follower, relation in mimics.items():
            parent = relation["parent"]
            pjid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, parent))
            fjid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower))
            qparent = float(data.qpos[int(model.jnt_qposadr[pjid])])
            set_joint(model, data, follower,
                      float(relation["multiplier"]) * qparent + float(relation["offset"]))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def highest_physical_gate(result: dict) -> str:
    events = {
        row.get("event") for row in result.get("events", [])
        if isinstance(row, dict)
    }
    ordered = (
        ("BOTTLE_SETTLED", "SETTLE"),
        ("OPEN_COMPLETE", "HAND_ACTUATION"),
        ("HOLD_VALIDATED", "FINGER_BOTTLE_CONTACT"),
        ("LOAD_READY", "LOAD_READY"),
        ("SHORT_LIFT_REACHED", "ISOLATED_PHYSICAL_LIFT_30MM"),
        ("POST_LIFT_HOLD_COMPLETE", "HOLD_1S"),
        ("RELEASE_COMPLETE", "PHYSICAL_RELEASE"),
    )
    highest = "PREFLIGHT" if result.get("preflight_status") != "PASS" else "WRENCH_FEASIBLE"
    for event_name, stage in ordered:
        if event_name in events:
            highest = stage
    return highest


def first_failed_gate(result: dict) -> str | None:
    gates = result.get("gate_status", {})
    return next((name for name in GATE_NAMES[:-1] if gates.get(name) != "PASS"), None)


def prepare_runtime_robot_xml(output: Path) -> Path:
    template = FIXTURE / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
    text = template.read_text(encoding="utf-8")
    replacements = {
        "@UNITREE_MUJOCO_G1_MESHES@": str(
            runner.UNITREE_MUJOCO / "unitree_robots/g1/meshes"
        ),
        "@UNITREE_ROS_G1_MESHES@": str(
            runner.UNITREE_ROS / "robots/g1_description/meshes"
        ),
    }
    for token, value in replacements.items():
        if token not in text:
            raise RuntimeError(f"MJCF asset path token is missing: {token}")
        text = text.replace(token, value)
    if "@UNITREE_" in text:
        raise RuntimeError("Unresolved upstream asset path remains in generated MJCF")
    runtime_xml = output / "compiled/g1_dfq_runtime_paths.xml"
    runtime_xml.write_text(text, encoding="utf-8")
    runner.ROBOT_XML = runtime_xml
    return runtime_xml


def normalized_model_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        str(runner.UNITREE_MUJOCO / "unitree_robots/g1/meshes"),
        "/tmp/robotsim-issue43-m0/checkouts/unitree_mujoco/unitree_robots/g1/meshes",
    )
    text = text.replace(
        str(runner.UNITREE_ROS / "robots/g1_description/meshes"),
        "/tmp/robotsim-issue43-m0/checkouts/unitree_ros_dex3/robots/g1_description/meshes",
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_report(path: Path, result: dict, command: str) -> None:
    settle = result.get("bottle_settle", {})
    load = result.get("load_build", {})
    stage_rows = result.get("lift_stages", [])
    stages = [str(row.get("stage")) for row in stage_rows if isinstance(row, dict)]
    gates = result.get("gate_evaluation", {})
    lines = [
        "# G1 DFQ Physical Grasp Implementation", "",
        f"Status: {result.get('status', 'UNKNOWN')}",
        f"Highest physical gate: {result.get('highest_physical_gate', 'not reached')}",
        f"Run ID: {result.get('run_id', 'unknown')}", "",
        "## Scope", "",
        "SIMULATION_ONLY_M0 evidence. This is not a real DFQ hardware equivalence claim.",
        "The accepted DFQ meshes, converter, wrist mounts, bottle/table geometry, source limits, 0.531 Nm effort cap, friction, mimic topology, and controller gains were preserved.",
        "The bottle remained free-jointed: no bottle weld/equality, mocap/follow-hand, or active-rollout qpos writes were used. Only the right hand was commanded.", "",
        "## Static Candidate", "",
        f"- Candidate: {result.get('candidate', 'C2')} (starting hypothesis; static wrench feasibility is not dynamic proof)",
        f"- RobotSim SHA: {result.get('RobotSim_sha')}",
        f"- Compiled model SHA-256: {result.get('compiled_model_sha256')}",
        f"- Geometry gate: {result.get('selected_candidate_configuration', {}).get('static_wrench_classification', 'see candidate_c2.json / wrench JSON')}",
        "- Candidate joint configuration and target contact points are stored in `m0_result.json` and the static C2 JSON.",
        "- The fresh static wrench allocation uses the candidate's measured MuJoCo surface normals and contact Jacobians; table support is excluded.", "",
        "## Dynamic Gates", "",
        f"- Bottle settle: {settle.get('status', 'not reached')}; elapsed={settle.get('elapsed_s', 'n/a')} s; dwell={settle.get('stable_dwell_s', 'n/a')} s; max linear/angular speed={settle.get('maximum_linear_speed_m_s', 'n/a')} m/s / {settle.get('maximum_angular_speed_rad_s', 'n/a')} rad/s.",
        f"- LOAD_BUILD: {load.get('status', 'not reached')}; details are in `load_build_trace.csv` and result JSON.",
        f"- Pickup stages reached: {', '.join(stages) if stages else 'none'}.",
        f"- Final lift: {result.get('lift_height_m', 'not reached')} m; 30 mm target; post-lift hold={result.get('post_lift_hold_seconds', 'not reached')} s.",
        f"- Peak source hand speed: {result.get('maximum_hand_joint_qvel_rad_s', 'not reached')} rad/s ({result.get('maximum_hand_joint_qvel_joint', 'n/a')}); peak applied simulation-only effort={result.get('maximum_simulation_only_driver_effort_nm', 'not reached')} Nm; peak mimic error={result.get('maximum_mimic_error_rad', 'not reached')} rad.",
        f"- Bottle qpos writes: {result.get('bottle_qpos_write_count', 0)}; runner write guard: {result.get('active_rollout_qpos_write_scan', 'not reached')}.",
        f"- First failure: {result.get('error', 'none')}.", "",
        "## Required Gates", "",
        "| Gate | Status | Reason |",
        "| --- | --- | --- |",
        *[
            f"| {name} | {gates.get(name, {}).get('status', 'NOT RUN')} | "
            f"{gates.get(name, {}).get('reason', 'no gate evidence')} |"
            for name in GATE_NAMES
        ],
        "",
        "## Evidence", "",
        "The run directory includes `m0_result.json`, traces, log, MP4, and stage PNGs when those stages were reached. No stage after the first failed physical gate is claimed.", "",
        "## Reproduction", "", command, "",
        "## Environment", "",
        f"- Python: {result.get('python_version')}",
        f"- MuJoCo: {result.get('mujoco_version')}",
        f"- Upstreams: {json.dumps(result.get('upstream', {}), sort_keys=True)}",
        f"- External runner SHA-256: {result.get('external_runner_sha256')}",
    ]
    (path / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path / "REPRODUCE.md").write_text("# Reproduction\n\n```powershell\n" + command + "\n```\n", encoding="utf-8")


def main() -> int:
    run_id = f"g1-dfq-physical-{time.time_ns()}-{os.getpid()}"
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    output = EVIDENCE / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "compiled").mkdir()
    command_env = {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "ROBOTSIM_MUJOCO_PYTHON": os.environ.get("ROBOTSIM_MUJOCO_PYTHON", sys.executable),
        "ROBOTSIM_EVIDENCE_ROOT": str(EVIDENCE),
        "ROBOTSIM_UNITREE_ROS_DIR": str(runner.UNITREE_ROS),
        "ROBOTSIM_UNITREE_MUJOCO_DIR": str(runner.UNITREE_MUJOCO),
    }
    command = " ".join(
        shlex.quote(f"{name}={value}") for name, value in command_env.items()
    ) + " ./scripts/run_g1_dfq_physical_grasp.sh"
    result = {
        "run_id": run_id,
        "candidate": "C2",
        "status": "PREFLIGHT",
        "passed": False,
        "highest_physical_gate": "C2 STATIC GEOMETRY / WRENCH ONLY",
        "physics_steps": 0,
        "reproduction_command": command,
        "RobotSim_sha": None,
        "RobotSim_branch": None,
        "RobotSim_dirty": None,
        "output_directory": str(output),
        "gate_status": {
            "HAND_ACTUATION": "NOT RUN",
            "FINGER_BOTTLE_CONTACT": "NOT RUN",
            "ISOLATED_PHYSICAL_LIFT_30MM": "NOT RUN",
            "HOLD_1S": "NOT RUN",
            "PHYSICAL_RELEASE": "NOT RUN",
            "FULL_WORKCELL_PICKUP": "NOT RUN",
        },
        "upstream": {
            "unitree_ros_sha": runner.UNITREE_ROS_SHA,
            "unitree_mujoco_sha": runner.UNITREE_MUJOCO_SHA,
            "humanoid_vla_sha": runner.HUMANOID_VLA_SHA,
            "humanoid_vla_role": "historical C2 candidate optimizer provenance only; not a runtime dependency",
        },
        "python_version": platform.python_version(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "candidate_static_run_id": None,
        "candidate_wrench_input_sha256": None,
        "candidate_wrench_result_sha256": None,
        "runner_sha256": runner.sha256(Path(__file__).resolve()),
        "external_runner_sha256": runner.sha256(Path(runner.__file__).resolve()),
        "physics_control": {
            "classification": "SIMULATION_ONLY_M0; no real DFQ hardware equivalence claim",
            "source_velocity_limit_rad_s": 0.5,
            "close_target_slew_rad_s": runner.CLOSE_TARGET_SLEW_RAD_S,
            "simulation_only_driver_effort_cap_nm": runner.SIM_ONLY_MAX_DRIVER_TORQUE_NM,
            "mimic_diagnostic_tolerance_rad": runner.MIMIC_DIAGNOSTIC_TOLERANCE_RAD,
            "mimic_manipulation_limit_rad": runner.MIMIC_MANIPULATION_LIMIT_RAD,
            "mimic_follower_actuators": 0,
            "mimic_follower_qpos_writes": 0,
            "bottle_weld_or_mocap": False,
            "bottle_qpos_writes_during_rollout": 0,
            "direct_arm_lift_after_contact": True,
            "load_build_mode": "fixed-wrist LOAD_BUILD skipped; bounded contact-force preload precedes commanded arm lift",
            "contact_force_preload": "SIMULATION_ONLY torque impedance plus evidence-derived measured-contact topology feedforward; not a LOAD_READY claim",
            "load_contact_reallocation": runner.LOAD_CONTACT_REALLOCATION,
            "close_target_source": "C2 whole-hand source-limited static optimizer; same 0.060 rad/s close slew",
        },
        "stage1_wrench_audit": {},
        "lift_arming": {},
        "load_capacity_audit": True,
        "load_transfer_only": False,
        "contact_only_diagnostic": False,
        "stop_after_load_build": False,
        "stage3_and_later_executed": True,
        "lift_target_height_m": 0.030,
        "post_lift_hold_seconds": 1.0,
        "preflight_status": "RUNNING",
        "full_workcell_integration_verified": False,
        "state_writes": "initial/reset setup only before rollout; active runner AST guard must report zero qpos writes",
        "bottle_qpos_write_count": 0,
    }
    write_json(output / "m0_result.json", result)
    (output / "run.log").write_text(
        f"run_id={run_id}\nstatus=PREFLIGHT\nphysics_steps=0\n"
        f"reproduction_command={command}\n", encoding="utf-8"
    )
    try:
        result.update({
            "RobotSim_sha": git(runner.REPO, "rev-parse", "HEAD"),
            "RobotSim_branch": git(runner.REPO, "branch", "--show-current"),
            "RobotSim_dirty": bool(git(runner.REPO, "status", "--porcelain")),
            "RobotSim_worktree": str(runner.REPO),
        })
        for path, expected in (
            (runner.UNITREE_ROS, runner.UNITREE_ROS_SHA),
            (runner.UNITREE_MUJOCO, runner.UNITREE_MUJOCO_SHA),
        ):
            observed = git(path, "rev-parse", "HEAD")
            dirty = git(path, "status", "--porcelain")
            if observed != expected or dirty:
                raise RuntimeError(f"pinned upstream mismatch/dirty checkout at {path}: {observed}, dirty={bool(dirty)}")
        if mujoco.__version__ != runner.MUJOCO_VERSION:
            raise RuntimeError(f"MuJoCo {runner.MUJOCO_VERSION} required, got {mujoco.__version__}")
        if runner.sha256(runner.URDF) != "e2b2d6bfe58f6b8d57b9f28df9ad9e5e5d0ea728f562e925b8b1482a902b6ff3":
            raise RuntimeError("Pinned official DFQ URDF SHA-256 mismatch")

        candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
        wrench = json.loads(WRENCH_PATH.read_text(encoding="utf-8"))
        result["candidate_static_run_id"] = candidate["run_id"]
        result["candidate_wrench_input_sha256"] = runner.sha256(
            C2_RUN / "candidate_c2_wrench_input.json"
        )
        result["candidate_wrench_result_sha256"] = runner.sha256(WRENCH_PATH)
        if candidate["static_gates"]["passed"] is not True:
            raise RuntimeError("C2 static geometry gate did not pass")
        if wrench.get("classification") != "A — NOMINAL GEOMETRY ACCEPTABLE":
            raise RuntimeError(f"C2 fresh wrench gate did not pass: {wrench.get('classification')}")
        if wrench.get("contact_force_group_gates", {}).get("effort_balanced_allocation", {}).get("thumb_plus_opposing_gate") != 1.0:
            raise RuntimeError("C2 wrench allocation lacks thumb plus opposing-digit support")

        hand.URDF = runner.URDF
        _, urdf_limits, mimics, _ = hand.parse_urdf()
        ranges = hand.derive_channel_ranges(urdf_limits, mimics)
        compiled_path = output / "compiled/dfq_c2_pickup_mj336.xml"
        runtime_robot_xml = prepare_runtime_robot_xml(output)
        result["runtime_robot_xml"] = str(runtime_robot_xml)
        result["runtime_robot_xml_sha256"] = runner.sha256(runtime_robot_xml)
        build = runner.build_scene(compiled_path, ranges, urdf_limits, mimics)
        model = build["model"]
        model_hash = runner.sha256(compiled_path)
        normalized_hash = normalized_model_sha256(compiled_path)
        expected_model_hash = candidate["source"]["compiled_model_sha256"]
        if normalized_hash != expected_model_hash:
            raise RuntimeError(
                "normalized C2 model hash changed: "
                f"{normalized_hash} != {expected_model_hash}"
            )
        result["compiled_model_sha256"] = model_hash
        result["normalized_compiled_model_sha256"] = normalized_hash
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
        for name, value in base["joint_state_qpos_rad"].items():
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if jid >= 0 and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                set_joint(model, data, name, float(value))
        candidate_arm = np.asarray([
            candidate["joint_configuration"][name]["q_rad"] for name in runner.ARM_NAMES
        ], dtype=float)
        for name in runner.ARM_NAMES:
            set_joint(model, data, name, candidate["joint_configuration"][name]["q_rad"])
        driver_targets = {
            channel: float(candidate["joint_configuration"][hand.channel_joint("R", suffix)]["q_rad"])
            for channel, suffix in hand.CHANNELS
        }
        for channel, suffix in hand.CHANNELS:
            set_joint(model, data, hand.channel_joint("R", suffix), driver_targets[channel])
        apply_mimics(model, data, mimics)
        bottle_qadr = runner.bottle_qpos_addresses(model)
        data.qpos[bottle_qadr] = np.r_[
            base["bottle_root_position_world_m"], base["bottle_root_quaternion_wxyz"]
        ]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        pregrasp_q = np.asarray(candidate["open_hand_approach"]["pregrasp_arm_q_rad"], dtype=float)
        approach_q = candidate_arm.copy()
        arm_qadr = runner.arm_qpos_addresses(model)
        data.qpos[arm_qadr] = approach_q
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site"))
        contact_site_pos = data.site_xpos[site_id].copy()
        contact_site_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        lift_q, lift_info = runner.plan_lift(
            model, data, site_id, approach_q, contact_site_pos, contact_site_rot, 0.030
        )
        data.qpos[arm_qadr] = pregrasp_q
        # Start the active run in the statically verified clear/open state.
        for channel, suffix in hand.CHANNELS:
            set_joint(model, data, hand.channel_joint("R", suffix),
                      hand.normalized_target("R", channel, 1.0, ranges))
        apply_mimics(model, data, mimics)
        mujoco.mj_forward(model, data)
        arm_contacts = [c for c in runner.bottle_contacts(model, data) if c["side"] != "other"]
        if arm_contacts:
            raise RuntimeError(f"open C2 pregrasp has robot/bottle contact: {arm_contacts}")

        meta = build
        meta["close_target_rad_by_channel"] = driver_targets
        load_capacity_targets = wrench
        result["compiled_model_sha256"] = model_hash
        result["selected_candidate_configuration"] = {
            "joint_configuration": candidate["joint_configuration"],
            "target_contact_geometry": candidate["joint_contact_geometry"],
            "static_wrench_classification": wrench["classification"],
            "wrench_allocation": wrench["effort_balanced_10pct_margin_allocation"],
            "pregrasp_arm_q_rad": pregrasp_q.tolist(),
            "approach_arm_q_rad": approach_q.tolist(),
            "lift_arm_q_rad": lift_q.tolist(),
            "lift_ik": lift_info,
        }
        result["stage1_wrench_audit"] = {
            "path": WRENCH_PATH.as_posix(),
            "sha256": runner.sha256(WRENCH_PATH),
            "source_run_id": wrench["run_id"],
            "physics_steps": wrench["physics_steps"],
            "bottle_mass_kg": wrench["bottle_mass_kg"],
            "bottle_weight_n": wrench["bottle_weight_n"],
            "configured_mu": wrench["configured_mu"],
            "allocation": wrench["effort_balanced_10pct_margin_allocation"],
            "robustness_summary": wrench["robustness_summary"],
        }
        result["status"] = "RUNNING"
        result["preflight_status"] = "PASS"
        result["highest_physical_gate"] = "WRENCH_FEASIBLE"
        result["active_rollout_qpos_write_scan"] = runner.ast_rollout_write_check()
        write_json(output / "preflight_plan.json", {
            "run_id": run_id,
            "candidate": "C2",
            "physics_steps": 0,
            "compiled_model_sha256": model_hash,
            "settle_clear_arm_target_rad": pregrasp_q.tolist(),
            "pregrasp_arm_target_rad": pregrasp_q.tolist(),
            "approach_arm_target_rad": approach_q.tolist(),
            "close_target_rad_by_channel": driver_targets,
            "close_target_slew_rad_s": runner.CLOSE_TARGET_SLEW_RAD_S,
            "lift_target_m": 0.030,
            "lift_arm_target_rad": lift_q.tolist(),
            "open_pregrasp_robot_bottle_contacts": arm_contacts,
            "reproduction_command": command,
        })

        outcome = runner.run_rollout(
            model, data, meta, output, result,
            pregrasp_q, pregrasp_q, approach_q, lift_q,
            ranges, urdf_limits, mimics,
            diagnostic_only=False,
            lift_height_m=0.030,
            post_lift_hold_seconds=1.0,
            load_transfer_only=False,
            load_capacity_targets=load_capacity_targets,
            stop_after_load_build=False,
            direct_arm_lift=True,
        )
        result = outcome["result"]
        result["physics_steps"] = int(outcome["physics_steps"])
        result["gate_evaluation"] = evaluate_gates(result)
        result["gate_status"] = {
            name: entry["status"] for name, entry in result["gate_evaluation"].items()
        }
        isolated_gates_passed = all(
            result["gate_status"][name] == "PASS" for name in GATE_NAMES[:-1]
        )
        result["passed"] = bool(result.get("passed")) and isolated_gates_passed
        result["status"] = "PASS" if result["passed"] else "FAIL"
        result["highest_physical_gate"] = highest_physical_gate(result)
        result["first_failed_gate"] = first_failed_gate(result)
        result["reproduction_command"] = command
        result["executed_runner_sha256"] = runner.sha256(Path(__file__).resolve())
        result["external_runner_sha256"] = runner.sha256(Path(runner.__file__).resolve())
        result["executed_runner_worktree"] = {
            "path": str(runner.REPO),
            "head": result["RobotSim_sha"],
            "branch": result["RobotSim_branch"],
            "dirty": result["RobotSim_dirty"],
        }
        write_json(output / "m0_result.json", result)
        write_report(output, result, command)
        (output / "run.log").write_text(
            f"run_id={run_id}\nstatus={result['status']}\nphysics_steps={result['physics_steps']}\n"
            f"highest_physical_gate={result['highest_physical_gate']}\nreproduction_command={command}\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "run_id": run_id,
            "status": result["status"],
            "passed": result.get("passed"),
            "physics_steps": result["physics_steps"],
            "highest_physical_gate": result["highest_physical_gate"],
            "lift_height_m": result.get("lift_height_m"),
            "peak_qvel_rad_s": result.get("maximum_hand_joint_qvel_rad_s"),
            "peak_effort_nm": result.get("maximum_simulation_only_driver_effort_nm"),
            "error": result.get("error"),
            "output": output.as_posix(),
        }, indent=2))
        return 0 if result.get("passed") else 2
    except Exception as exc:
        result["status"] = "FAIL"
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["gate_evaluation"] = evaluate_gates(result)
        result["gate_status"] = {
            name: entry["status"] for name, entry in result["gate_evaluation"].items()
        }
        result["highest_physical_gate"] = highest_physical_gate(result)
        result["first_failed_gate"] = (
            first_failed_gate(result) if result.get("preflight_status") == "PASS"
            else "PREFLIGHT"
        )
        result["first_failed_gate_reason"] = result["error"]
        result["executed_runner_sha256"] = runner.sha256(Path(__file__).resolve())
        result["external_runner_sha256"] = runner.sha256(Path(runner.__file__).resolve())
        write_json(output / "m0_result.json", result)
        write_report(output, result, command)
        (output / "run.log").write_text(
            f"run_id={run_id}\nstatus=FAIL\nerror={result['error']}\nreproduction_command={command}\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "run_id": run_id,
            "status": "FAIL",
            "error": result["error"],
            "output": output.as_posix(),
        }, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
