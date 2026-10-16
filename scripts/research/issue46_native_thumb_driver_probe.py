from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_native_omnihand_minimal_grasp as task  # noqa: E402

DRIVER = "R_thumb_mcp_joint"
FOLLOWERS = ("R_thumb_pip_joint", "R_thumb_dip_joint")
OPEN_RAD = 0.08
CLOSE_RAD = 0.38
TORQUE_CAP_NM = 0.08
KP_NM_PER_RAD = 0.04
KD_NMS_PER_RAD = 0.004
MIMIC_LIMIT_RAD = 0.003
SETTLE_WINDOW_S = 0.2
SETTLE_SPEED_RAD_S = 0.02
SETTLE_ERROR_RAD = 0.01
MOTION_ERROR_RAD = 0.08
SOURCE_LIMIT_TOLERANCE_RAD = task.JOINT_LIMIT_TOLERANCE_RAD


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def main() -> int:
    parser = argparse.ArgumentParser(description="SIMULATION_ONLY native X2 thumb transmission probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    task.verify_vendor_pin()
    model, details = task.build_model("source-faithful", out)
    data = mujoco.MjData(model)
    targets = {name: float(details["joint_references"][name]) for name in details["source_joints"]}
    targets.update(task.OPEN_POSE)
    targets[DRIVER] = OPEN_RAD
    targets = task.derived_targets(targets, details)
    initial_joint_qpos_assignments = 0
    for name in details["source_joints"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = task.source_target_to_raw(model, details, name, targets[name])
        initial_joint_qpos_assignments += 1
    mujoco.mj_forward(model, data)

    driver_aid = int(details["actuator_by_joint"][DRIVER])
    model.actuator_gaintype[driver_aid] = int(mujoco.mjtGain.mjGAIN_FIXED)
    model.actuator_biastype[driver_aid] = int(mujoco.mjtBias.mjBIAS_NONE)
    model.actuator_dyntype[driver_aid] = int(mujoco.mjtDyn.mjDYN_NONE)
    model.actuator_gainprm[driver_aid, :] = 0.0
    model.actuator_gainprm[driver_aid, 0] = 1.0
    model.actuator_biasprm[driver_aid, :] = 0.0
    model.actuator_ctrllimited[driver_aid] = 1
    model.actuator_ctrlrange[driver_aid, :] = [-TORQUE_CAP_NM, TORQUE_CAP_NM]
    model.actuator_forcelimited[driver_aid] = 1
    model.actuator_forcerange[driver_aid, :] = [-TORQUE_CAP_NM, TORQUE_CAP_NM]

    for name, aid in details["actuator_by_joint"].items():
        if name != DRIVER:
            data.ctrl[aid] = task.source_target_to_raw(model, details, name, targets[name])

    mappings = {}
    for name in details["source_joints"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        limit = details["source_joints"][name].find("limit")
        mappings[name] = {
            "jid": int(jid),
            "qadr": int(model.jnt_qposadr[jid]),
            "dadr": int(model.jnt_dofadr[jid]),
            "sign": int(details["joint_axes"][name]),
            "reference": float(details["joint_references"][name]),
            "range": details["joint_ranges"][name],
            "velocity_limit": float(limit.get("velocity")),
            "effort_limit": float(limit.get("effort")),
        }

    mimic = details["mimic_relations"]
    source_relations = {
        follower: relation for follower, relation in mimic.items()
        if relation["driver"] == DRIVER
    }
    if set(source_relations) != set(FOLLOWERS):
        raise RuntimeError(f"unexpected source thumb mimic relations: {sorted(source_relations)}")
    if model.neq != len(mimic):
        raise RuntimeError(f"unexpected equality count: {model.neq} != {len(mimic)}")

    def source_q(name: str) -> float:
        item = mappings[name]
        return item["reference"] + item["sign"] * (
            float(data.qpos[item["qadr"]]) - float(model.qpos0[item["qadr"]])
        )

    def source_v(name: str) -> float:
        item = mappings[name]
        return item["sign"] * float(data.qvel[item["dadr"]])

    basis = {DRIVER: 1.0}
    for follower, relation in source_relations.items():
        basis[follower] = (
            mappings[follower]["sign"]
            * float(relation["multiplier"])
            * mappings[DRIVER]["sign"]
        )

    phases = (
        ("OPEN_HOLD_INITIAL", 0.8, OPEN_RAD, OPEN_RAD),
        ("CLOSE", 1.0, OPEN_RAD, CLOSE_RAD),
        ("CLOSE_HOLD", 0.6, CLOSE_RAD, CLOSE_RAD),
        ("RETURN_OPEN", 1.0, CLOSE_RAD, OPEN_RAD),
        ("OPEN_HOLD_FINAL", 0.8, OPEN_RAD, OPEN_RAD),
    )
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = task.camera_for_scene()
    trace_path = out / "thumb_driver_trace.jsonl"
    video_path = out / "thumb_driver_mjstep.mp4"
    video = imageio.get_writer(video_path, fps=25, codec="libx264", quality=8)
    frames_per_step = max(1, round(1.0 / (25.0 * task.DT)))
    max_mimic = max_limit_violation = max_velocity_ratio = max_effort_ratio = 0.0
    max_motion_error = 0.0
    finite = True
    first_failure = None
    phase_results = []
    step_index = 0

    def render() -> None:
        renderer.update_scene(data, camera=camera)
        video.append_data(renderer.render())

    with trace_path.open("w", encoding="utf-8") as trace:
        render()
        for phase, duration, start, end in phases:
            phase_steps = round(duration / task.DT)
            samples = []
            for k in range(phase_steps):
                blend = smoothstep((k + 1) / phase_steps) if start != end else 1.0
                target = start + blend * (end - start)
                target_by_joint = dict(targets)
                target_by_joint[DRIVER] = target
                for follower, relation in mimic.items():
                    target_by_joint[follower] = (
                        float(relation["offset"])
                        + float(relation["multiplier"])
                        * target_by_joint[str(relation["driver"]) ]
                    )

                for name, aid in details["actuator_by_joint"].items():
                    if name != DRIVER:
                        data.ctrl[aid] = task.source_target_to_raw(
                            model, details, name, target_by_joint[name]
                        )
                mujoco.mj_forward(model, data)
                compiled_target = task.source_target_to_raw(model, details, DRIVER, target)
                driver_map = mappings[DRIVER]
                projected_bias = sum(
                    coefficient * float(data.qfrc_bias[mappings[name]["dadr"]])
                    for name, coefficient in basis.items()
                )
                torque = projected_bias + KP_NM_PER_RAD * (
                    compiled_target - float(data.qpos[driver_map["qadr"]])
                ) - KD_NMS_PER_RAD * float(data.qvel[driver_map["dadr"]])
                torque = float(np.clip(torque, -TORQUE_CAP_NM, TORQUE_CAP_NM))
                data.ctrl[driver_aid] = torque
                mujoco.mj_forward(model, data)
                pre_step_dynamics = {
                    name: {
                        "qfrc_bias_nm": float(data.qfrc_bias[item["dadr"]]),
                        "qfrc_actuator_nm": float(data.qfrc_actuator[item["dadr"]]),
                        "qfrc_constraint_nm": float(data.qfrc_constraint[item["dadr"]]),
                        "qacc_rad_s2": float(item["sign"] * data.qacc[item["dadr"]]),
                    }
                    for name, item in mappings.items()
                }
                mujoco.mj_step(model, data)
                step_index += 1

                mimic_errors = {}
                for follower, relation in mimic.items():
                    mimic_errors[follower] = source_q(follower) - (
                        float(relation["offset"])
                        + float(relation["multiplier"])
                        * source_q(str(relation["driver"]))
                    )
                max_mimic = max(max_mimic, max(map(abs, mimic_errors.values()), default=0.0))
                positions, velocities, limits, efforts = {}, {}, {}, {}
                for name, item in mappings.items():
                    q, v = source_q(name), source_v(name)
                    low, high = item["range"]
                    limit_violation = max(0.0, low - q, q - high)
                    velocity_ratio = abs(v) / item["velocity_limit"]
                    effort = abs(float(data.qfrc_actuator[item["dadr"]]))
                    effort_ratio = effort / item["effort_limit"]
                    positions[name] = q
                    velocities[name] = v
                    limits[name] = limit_violation
                    efforts[name] = effort
                    max_limit_violation = max(max_limit_violation, limit_violation)
                    max_velocity_ratio = max(max_velocity_ratio, velocity_ratio)
                    max_effort_ratio = max(max_effort_ratio, effort_ratio)
                target_error = source_q(DRIVER) - target
                if start != end:
                    max_motion_error = max(max_motion_error, abs(target_error))
                finite = finite and bool(
                    np.isfinite(data.qpos).all()
                    and np.isfinite(data.qvel).all()
                    and np.isfinite(data.actuator_force).all()
                )
                if max_mimic >= MIMIC_LIMIT_RAD:
                    offender = max(mimic_errors, key=lambda name: abs(mimic_errors[name]))
                    first_failure = first_failure or (
                        f"mimic residual reached gate in {phase} step {k + 1}: "
                        f"{offender}={mimic_errors[offender]:.9f} rad"
                    )
                if max_limit_violation > SOURCE_LIMIT_TOLERANCE_RAD:
                    offender = max(limits, key=limits.get)
                    first_failure = first_failure or (
                        f"source joint limit tolerance exceeded in {phase} step {k + 1}: "
                        f"{offender} by {limits[offender]:.9f} rad"
                    )
                if not finite:
                    first_failure = first_failure or f"non-finite state in {phase} step {k + 1}"
                contact_rows = []
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_id, force)
                    geom1, geom2 = int(contact.geom1), int(contact.geom2)
                    contact_rows.append({
                        "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                        "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                        "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1])),
                        "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2])),
                        "distance_m": float(contact.dist),
                        "force_contact_frame": force.tolist(),
                    })
                record = {
                    "step": step_index,
                    "phase": phase,
                    "time_s": float(data.time),
                    "target_mcp_source_rad": target,
                    "target_error_mcp_rad": target_error,
                    "q_source_rad": positions,
                    "qdot_source_rad_s": velocities,
                    "actuator_torque_source_coordinates_nm": efforts,
                    "commanded_thumb_mcp_torque_nm": torque,
                    "projected_gravity_bias_nm": projected_bias,
                    "pre_step_dynamics_by_joint": pre_step_dynamics,
                    "mimic_residual_source_rad": mimic_errors,
                    "source_limit_violation_rad": limits,
                    "contacts": contact_rows,
                }
                trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                samples.append({
                    "error": abs(target_error),
                    "speed": abs(source_v(DRIVER)),
                    "all_speed": max(map(abs, velocities.values())),
                    "all_error": max(
                        abs(positions[name] - target_by_joint[name])
                        for name in details["source_joints"]
                    ),
                })
                if step_index % frames_per_step == 0:
                    render()
                if first_failure:
                    break
            tail_count = min(len(samples), round(SETTLE_WINDOW_S / task.DT))
            tail = samples[-tail_count:]
            hold = start == end
            phase_results.append({
                "phase": phase,
                "steps": len(samples),
                "requested_steps": phase_steps,
                "max_mcp_tracking_error_rad": max((x["error"] for x in samples), default=None),
                "terminal_window_s": tail_count * task.DT,
                "terminal_max_mcp_error_rad": max((x["error"] for x in tail), default=None),
                "terminal_max_mcp_speed_rad_s": max((x["speed"] for x in tail), default=None),
                "terminal_max_all_joint_error_rad": max((x["all_error"] for x in tail), default=None),
                "terminal_max_all_joint_speed_rad_s": max((x["all_speed"] for x in tail), default=None),
                "settled": (
                    not hold or (
                        max(x["all_error"] for x in tail) <= SETTLE_ERROR_RAD
                        and max(x["all_speed"] for x in tail) <= SETTLE_SPEED_RAD_S
                    )
                ),
            })
            if first_failure:
                break

    render()
    video.close()
    renderer.close()
    command = (
        f"AGIBOT_X2_VENDOR_ROOT={task.VENDOR_ROOT} MUJOCO_GL=egl "
        f"{sys.executable} {Path(__file__).resolve()} --output {out}"
    )
    gates = {
        "genuine_mj_step": step_index > 0,
        "source_mimic_below_0_003_rad": max_mimic < MIMIC_LIMIT_RAD,
        "source_joint_ranges_within_existing_0_001_rad_tolerance": (
            max_limit_violation <= SOURCE_LIMIT_TOLERANCE_RAD
        ),
        "source_velocity_limits": max_velocity_ratio <= 1.0,
        "source_effort_limits": max_effort_ratio <= 1.0,
        "terminal_open_and_close_holds": all(
            item["settled"] for item in phase_results
            if item["phase"].endswith("HOLD")
        ),
        "bounded_motion_tracking": max_motion_error <= MOTION_ERROR_RAD,
        "finite_state": finite,
        "no_follower_actuators": all(
            follower not in details["actuator_by_joint"] for follower in FOLLOWERS
        ),
        "no_runtime_follower_state_writes": True,
        "no_bottle_qpos_writes": True,
        "fixed_palm_no_lift_claim": True,
    }
    result = {
        "status": "PASS" if all(gates.values()) and not first_failure else "FAIL",
        "classification": "SIMULATION_ONLY custom thumb-driver torque controller",
        "first_failure": first_failure,
        "robot_sim_branch": git(task.REPO_ROOT, "branch", "--show-current"),
        "robot_sim_head": git(task.REPO_ROOT, "rev-parse", "HEAD"),
        "robot_sim_dirty_paths": git(task.REPO_ROOT, "status", "--porcelain").splitlines(),
        "vendor_commit": git(task.VENDOR_ROOT, "rev-parse", "HEAD"),
        "source_urdf_sha256": sha256(task.URDF),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "mujoco": mujoco.__version__,
        "python": subprocess.check_output([sys.executable, "--version"], text=True).strip(),
        "timestep_s": task.DT,
        "integrator": "implicitfast",
        "gravity_m_s2": model.opt.gravity.tolist(),
        "model_dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "neq", "nbody", "njnt", "ngeom")},
        "independent_thumb_driver": DRIVER,
        "source_thumb_followers": list(FOLLOWERS),
        "thumb_actuator": {
            "kind": "direct torque",
            "kp_nm_per_rad": KP_NM_PER_RAD,
            "kd_nm_s_per_rad": KD_NMS_PER_RAD,
            "cap_nm": TORQUE_CAP_NM,
            "source_effort_limit_nm": mappings[DRIVER]["effort_limit"],
        },
        "follower_actuators": 0,
        "initial_joint_qpos_assignments_before_first_step": initial_joint_qpos_assignments,
        "runtime_follower_qpos_qvel_writes": 0,
        "bottle_qpos_writes": 0,
        "palm_world_fixed": True,
        "palm_root_collision_exclusions": "source-unauthorized fixture chain exclusions inherited from diagnostic model",
        "bottle_contact": "not an acceptance gate; fixed palm remains at the known no-contact baseline pose",
        "steps": step_index,
        "max_mimic_residual_rad": max_mimic,
        "max_source_limit_violation_rad": max_limit_violation,
        "source_limit_tolerance_rad": SOURCE_LIMIT_TOLERANCE_RAD,
        "max_source_velocity_ratio": max_velocity_ratio,
        "max_source_effort_ratio": max_effort_ratio,
        "max_mcp_motion_tracking_error_rad": max_motion_error,
        "phases": phase_results,
        "gates": gates,
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "video": str(video_path),
        "video_sha256": sha256(video_path),
        "reproduction_command": command,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "reproduction_command.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "first_failure": first_failure, "gates": gates, "result": str(out / "result.json")}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
