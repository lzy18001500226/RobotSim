#!/usr/bin/env python3
"""One controlled pyramidal-vs-elliptic contact diagnostic for Issue #46.

This is a SIMULATION_ONLY diagnostic on the independent ideal parallel-gripper
fixture. It is not an X2 model or a grasp acceptance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from scripts.research.issue46_ideal_parallel_gripper_baseline import (
    CYLINDER_BODY,
    CYLINDER_GEOM,
    LEFT_JOINT,
    LEFT_PAD_BODY,
    LIFT_JOINT,
    RIGHT_JOINT,
    RIGHT_PAD_BODY,
    actuator_id,
    body_id,
    joint_id,
)


EXPECTED_SOURCE_XML_SHA256 = "b44a29746c1a7871236f47b96ca6d086ea5f6533ebcbe8014dea6490a2ebf24e"
EXPECTED_SOURCE_TRACE_SHA256 = "3bed5cdb90f651e05fb8560a145fd71e23eccb5f626eb9825e5a05468463e770"
EXPECTED_SOURCE_RUN_SHA = "0510a742c698efa29c7fe0202d225524fed3ab0d"
EXPECTED_MUJOCO_VERSION = "3.3.6"
SOURCE_PHASE = "BILATERAL_HOLD_ON_TABLE"
PAD_TARGET_M = 0.022
LIFT_BASE_TARGET_M = 0.010
DIAGNOSTIC_LIFT_M = 0.010
RAMP_DURATION_S = 0.50
HOLD_DURATION_S = 0.50


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inject_cone(xml_text: str, cone: str) -> str:
    matches = list(re.finditer(r"<option\b[^>]*/>", xml_text))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one self-closing option tag, found {len(matches)}")
    tag = matches[0].group(0)
    if re.search(r"\bcone\s*=", tag):
        raise RuntimeError("Pinned source scene already specifies a cone; refusing a second edit")
    replacement = tag[:-2] + f' cone="{cone}"/>'
    return xml_text[: matches[0].start()] + replacement + xml_text[matches[0].end() :]


def joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, name)])


def joint_dof_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, name)])


def body_twist(model: mujoco.MjModel, data: mujoco.MjData, bid: int) -> np.ndarray:
    twist = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, twist, 0)
    return twist


def restore_state(
    model: mujoco.MjModel,
    source_row: dict[str, Any],
) -> tuple[mujoco.MjData, dict[str, Any]]:
    """Restore the logged post-step HOLD state, then recompute derived state."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    bottle = source_row["bottle"]
    free_joint_id = joint_id(model, "canonical_test_cylinder_free")
    qadr = int(model.jnt_qposadr[free_joint_id])
    data.qpos[qadr : qadr + 7] = np.asarray(
        bottle["position_world_m"] + bottle["quaternion_wxyz"], dtype=np.float64
    )
    free_dadr = int(model.jnt_dofadr[free_joint_id])
    bid = body_id(model, CYLINDER_BODY)

    # Map the recorded world-frame body twist into the compiled free-joint
    # generalized coordinates through MuJoCo's Jacobian, rather than assuming
    # a quaternion/free-joint velocity ordering or frame convention.
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[bid].copy(), bid)
    twist_jac = np.vstack((jacr[:, free_dadr : free_dadr + 6], jacp[:, free_dadr : free_dadr + 6]))
    target_twist = np.asarray(
        bottle["angular_velocity_world_rad_s"] + bottle["linear_velocity_world_m_s"],
        dtype=np.float64,
    )
    data.qvel[free_dadr : free_dadr + 6] = np.linalg.solve(twist_jac, target_twist)

    for joint_name, actuator_name, value_key in (
        (LEFT_JOINT, "left_pad_bounded_servo", "left"),
        (RIGHT_JOINT, "right_pad_bounded_servo", "right"),
        (LIFT_JOINT, "bounded_vertical_carriage_servo", "lift"),
    ):
        actuator_state = source_row["joints_actuators"][value_key]
        data.qpos[joint_qpos_address(model, joint_name)] = float(actuator_state["qpos"])
        data.qvel[joint_dof_address(model, joint_name)] = float(actuator_state["qvel"])
        data.ctrl[actuator_id(model, actuator_name)] = float(
            source_row["command"]["pad_target_q_m"]
            if value_key != "lift"
            else source_row["command"]["lift_target_q_m"]
        )

    data.time = float(source_row["time_s"])
    data.qacc_warmstart[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)

    rebuilt_twist = body_twist(model, data, bid)
    twist_error = float(np.max(np.abs(rebuilt_twist - target_twist)))
    position_error = float(
        np.max(np.abs(data.xpos[bid] - np.asarray(bottle["position_world_m"], dtype=np.float64)))
    )
    quaternion_error = float(
        np.max(np.abs(data.xquat[bid] - np.asarray(bottle["quaternion_wxyz"], dtype=np.float64)))
    )
    if max(twist_error, position_error, quaternion_error) > 2e-8:
        raise RuntimeError(
            "Restored state does not reproduce the saved bottle pose/twist: "
            f"position={position_error:.3g}, quaternion={quaternion_error:.3g}, twist={twist_error:.3g}"
        )

    details = {
        "source_trace_step": int(source_row["step"]),
        "source_trace_time_s": float(source_row["time_s"]),
        "source_phase": source_row["phase"],
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
        "state_sha256": sha256_bytes(
            np.asarray(data.qpos, dtype="<f8").tobytes()
            + np.asarray(data.qvel, dtype="<f8").tobytes()
            + np.asarray(data.ctrl, dtype="<f8").tobytes()
        ),
        "rebuild_validation": {
            "bottle_position_max_abs_error_m": position_error,
            "bottle_quaternion_max_abs_error": quaternion_error,
            "bottle_world_twist_max_abs_error": twist_error,
            "forward_recomputed": True,
            "warmstart_cleared": True,
            "external_applied_forces_zero": True,
        },
    }
    return data, details


def enum_names(enum_type: Any) -> dict[int, str]:
    members = getattr(enum_type, "__members__", {})
    return {int(value): str(name) for name, value in members.items()}


def point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body: int, point: np.ndarray
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point, body)
    return jacp @ data.qvel


def contact_constraint_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_index: int,
    type_names: dict[int, str],
    state_names: dict[int, str],
) -> list[dict[str, Any]]:
    if not all(hasattr(data, name) for name in ("efc_id", "efc_type", "efc_state", "efc_force")):
        return []
    contact_types = {
        int(getattr(mujoco.mjtConstraint, name))
        for name in (
            "mjCNSTR_CONTACT_FRICTIONLESS",
            "mjCNSTR_CONTACT_PYRAMIDAL",
            "mjCNSTR_CONTACT_ELLIPTIC",
        )
        if hasattr(mujoco.mjtConstraint, name)
    }
    rows = []
    for row in range(int(data.nefc)):
        if int(data.efc_id[row]) != contact_index or int(data.efc_type[row]) not in contact_types:
            continue
        state_code = int(data.efc_state[row])
        state_name = state_names.get(state_code, str(state_code))
        rows.append(
            {
                "efc_row": row,
                "type": type_names.get(int(data.efc_type[row]), str(int(data.efc_type[row]))),
                "state": state_name,
                "state_code": state_code,
                "nonzero_cost_state": "SATISFIED" not in state_name.upper(),
                "force": float(data.efc_force[row]),
            }
        )
    return rows


def contact_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    type_names: dict[int, str],
    state_names: dict[int, str],
) -> list[dict[str, Any]]:
    bottle_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CYLINDER_GEOM))
    bottle_body = body_id(model, CYLINDER_BODY)
    records = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or "world_geom"
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or "world_geom"
        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
        point = np.asarray(contact.pos, dtype=np.float64).copy()
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, local)
        world_on_geom2 = frame.T @ local[:3]
        world_tangent_on_geom2 = frame.T @ np.asarray([0.0, local[1], local[2]])

        if bottle_geom in (geom1, geom2):
            bottle_is_geom2 = geom2 == bottle_geom
            other_geom = geom1 if bottle_is_geom2 else geom2
            other_body = body1 if bottle_is_geom2 else body2
            force_on_bottle = world_on_geom2 if bottle_is_geom2 else -world_on_geom2
            tangent_on_bottle = world_tangent_on_geom2 if bottle_is_geom2 else -world_tangent_on_geom2
            normal_on_bottle = frame[0] if bottle_is_geom2 else -frame[0]
            other_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or "world"
            if other_body_name == "m0_table":
                support_class = "table"
            elif other_body_name in (LEFT_PAD_BODY, RIGHT_PAD_BODY):
                support_class = "hand_pad"
            else:
                support_class = "other"

            relative_velocity = point_velocity(model, data, bottle_body, point) - point_velocity(
                model, data, other_body, point
            )
            normal_component = float(np.dot(relative_velocity, frame[0]))
            relative_tangent = relative_velocity - normal_component * frame[0]
            friction = np.asarray(contact.friction, dtype=np.float64)
            normal_force = max(0.0, float(local[0]))
            if normal_force > 1e-12 and friction[0] > 0.0 and friction[1] > 0.0:
                ratios = np.asarray(
                    [local[1] / (friction[0] * normal_force), local[2] / (friction[1] * normal_force)],
                    dtype=np.float64,
                )
                pyramid_tangent = float(np.sum(np.abs(ratios)))
                elliptic_tangent = float(np.linalg.norm(ratios))
                torsional = (
                    abs(float(local[3])) / (float(friction[2]) * normal_force)
                    if int(contact.dim) >= 4 and friction[2] > 0.0
                    else 0.0
                )
            else:
                pyramid_tangent = elliptic_tangent = torsional = 0.0
            pyramid_full = pyramid_tangent + torsional
            elliptic_full = math.sqrt(elliptic_tangent**2 + torsional**2)
            cone_specific = elliptic_full if not mujoco.mj_isPyramidal(model) else pyramid_full
            records.append(
                {
                    "contact_index": index,
                    "pair": [geom1_name, geom2_name],
                    "body_pair": [body1_name, body2_name],
                    "support_class": support_class,
                    "other_body": other_body_name,
                    "contact_position_world_m": point.tolist(),
                    "contact_frame_rows_world": frame.tolist(),
                    "normal_on_bottle_world": normal_on_bottle.tolist(),
                    "contact_force_local_N_Nm": local.tolist(),
                    "force_on_bottle_world_N": force_on_bottle.tolist(),
                    "tangential_force_on_bottle_world_N": tangent_on_bottle.tolist(),
                    "normal_force_N": normal_force,
                    "tangential_force_components_local_N": [float(local[1]), float(local[2])],
                    "tangential_force_magnitude_N": float(np.linalg.norm(local[1:3])),
                    "relative_velocity_world_m_s": relative_velocity.tolist(),
                    "relative_tangential_velocity_world_m_s": relative_tangent.tolist(),
                    "relative_tangential_speed_m_s": float(np.linalg.norm(relative_tangent)),
                    "friction_coefficients_compiled": friction.tolist(),
                    "condim": int(contact.dim),
                    "cone_utilization": {
                        "pyramidal_tangent_l1": pyramid_tangent,
                        "elliptic_tangent_l2": elliptic_tangent,
                        "pyramidal_full_condim4_l1": pyramid_full,
                        "elliptic_full_condim4_l2": elliptic_full,
                        "active_model_full_condim4_measure": cone_specific,
                        "torsional_ratio_separate": torsional,
                        "not_a_solver_active_flag": True,
                    },
                    "solref": np.asarray(contact.solref, dtype=np.float64).tolist(),
                    "solimp": np.asarray(contact.solimp, dtype=np.float64).tolist(),
                    "distance_m": float(contact.dist),
                    "penetration_m": float(max(0.0, -contact.dist)),
                    "efc_address": int(getattr(contact, "efc_address", -1)),
                    "contact_constraint_rows": contact_constraint_rows(
                        model, data, index, type_names, state_names
                    ),
                }
            )
        else:
            records.append(
                {
                    "contact_index": index,
                    "pair": [geom1_name, geom2_name],
                    "body_pair": [body1_name, body2_name],
                    "support_class": "non_bottle",
                    "contact_position_world_m": point.tolist(),
                    "normal_force_N": max(0.0, float(local[0])),
                    "distance_m": float(contact.dist),
                    "penetration_m": float(max(0.0, -contact.dist)),
                    "efc_address": int(getattr(contact, "efc_address", -1)),
                    "contact_constraint_rows": contact_constraint_rows(
                        model, data, index, type_names, state_names
                    ),
                }
            )
    return records


def actuator_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pad: float,
    target_lift: float,
) -> dict[str, Any]:
    result = {}
    for key, joint, actuator, target, cap in (
        ("left", LEFT_JOINT, "left_pad_bounded_servo", target_pad, 8.0),
        ("right", RIGHT_JOINT, "right_pad_bounded_servo", target_pad, 8.0),
        ("lift", LIFT_JOINT, "bounded_vertical_carriage_servo", target_lift, 25.0),
    ):
        aid = actuator_id(model, actuator)
        jid = joint_id(model, joint)
        qa = int(model.jnt_qposadr[jid])
        da = int(model.jnt_dofadr[jid])
        force = float(data.actuator_force[aid])
        result[key] = {
            "joint": joint,
            "qpos": float(data.qpos[qa]),
            "qvel": float(data.qvel[da]),
            "target_q": float(target),
            "tracking_error_q_target_minus_actual": float(target - data.qpos[qa]),
            "actuator_force_N": force,
            "generalized_actuator_force_N": float(data.qfrc_actuator[da]),
            "effort_limit_N": cap,
            "saturated": abs(force) >= cap - 1e-7,
            "source_limit_margin": [
                float(data.qpos[qa] - model.jnt_range[jid, 0]),
                float(model.jnt_range[jid, 1] - data.qpos[qa]),
            ],
        }
    return result


def solver_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    raw_niter = np.asarray(data.solver_niter, dtype=np.int64).reshape(-1)
    max_niter = int(np.max(raw_niter, initial=0))
    stats = []
    # MuJoCo preallocates a large solver-stat buffer. Only the initial state
    # plus iterations actually performed are relevant for this step.
    for index, stat in enumerate(data.solver[: max_niter + 1]):
        row = {
            "iteration_stat_index": index,
            "niter": int(getattr(stat, "niter", 0)),
            "nactive": int(getattr(stat, "nactive", 0)),
            "nj": int(getattr(stat, "nj", 0)),
            "gradient": float(getattr(stat, "gradient", 0.0)),
            "improvement": float(getattr(stat, "improvement", 0.0)),
            "lineslope": float(getattr(stat, "lineslope", 0.0)),
        }
        stats.append(row)
    nonzero_niter = [
        {"island_index": int(index), "niter": int(value)}
        for index, value in enumerate(raw_niter)
        if value != 0
    ]
    raw_nnz = np.asarray(data.solver_nnz, dtype=np.int64).reshape(-1)
    warnings = [
        {"number": int(warning.number), "lastinfo": int(warning.lastinfo)}
        for warning in data.warning
        if int(warning.number) != 0
    ]
    return {
        "solver_niter": [entry["niter"] for entry in nonzero_niter],
        "solver_niter_nonzero_by_island_index": nonzero_niter,
        "solver_niter_vector_length": int(raw_niter.size),
        "solver_niter_max": int(np.max(raw_niter, initial=0)),
        "solver_nnz_nonzero": [int(value) for value in raw_nnz if value != 0],
        "solver_nnz_vector_length": int(raw_nnz.size),
        "solver_iteration_stats_initial_and_performed": stats,
        "solver_fwdinv": np.asarray(data.solver_fwdinv, dtype=np.float64).tolist(),
        "warnings": warnings,
        "nconstraint_rows": int(data.nefc),
    }


def sample_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    elapsed: float,
    target_lift: float,
    type_names: dict[int, str],
    state_names: dict[int, str],
) -> dict[str, Any]:
    contacts = contact_snapshot(model, data, type_names, state_names)
    bottle_contacts = [row for row in contacts if row["support_class"] != "non_bottle"]
    hand = [row for row in bottle_contacts if row["support_class"] == "hand_pad"]
    table = [row for row in bottle_contacts if row["support_class"] == "table"]
    hand_bodies = sorted({row["other_body"] for row in hand})
    bottle_bid = body_id(model, CYLINDER_BODY)
    lift_bid = body_id(model, "ideal_gripper_carriage")
    return {
        "time_s": float(data.time),
        "elapsed_from_hold_state_s": elapsed,
        "stage": "VERTICAL_SHEAR_RAMP" if elapsed < RAMP_DURATION_S else "VERTICAL_SHEAR_HOLD",
        "cone": "pyramidal" if mujoco.mj_isPyramidal(model) else "elliptic",
        "contact_force_evaluation_time_s": float(data.time),
        "contact_count_all": int(data.ncon),
        "bottle_contact_count": len(bottle_contacts),
        "hand_contact_count": len(hand),
        "hand_contact_bodies": hand_bodies,
        "table_contact_count": len(table),
        "bottle_world_position_m": data.xpos[bottle_bid].tolist(),
        "bottle_world_linear_velocity_m_s": body_twist(model, data, bottle_bid)[3:].tolist(),
        "carriage_world_position_m": data.xpos[lift_bid].tolist(),
        "hand_signed_world_vertical_support_N": math.fsum(
            float(row["force_on_bottle_world_N"][2]) for row in hand
        ),
        "table_signed_world_vertical_support_N": math.fsum(
            float(row["force_on_bottle_world_N"][2]) for row in table
        ),
        "actuators": actuator_snapshot(model, data, PAD_TARGET_M, target_lift),
        "solver": solver_snapshot(model, data),
        "contacts": contacts,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ramp = [row for row in samples if row["stage"] == "VERTICAL_SHEAR_RAMP"]
    hold = [row for row in samples if row["stage"] == "VERTICAL_SHEAR_HOLD"]

    def stats(values: Any) -> dict[str, float | None]:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
        }

    def stage_contact_statistics(stage_samples: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for side in (LEFT_PAD_BODY, RIGHT_PAD_BODY):
            contacts = [
                contact
                for sample in stage_samples
                for contact in sample["contacts"]
                if contact.get("support_class") == "hand_pad" and contact["other_body"] == side
            ]
            constraint_rows = [
                row for contact in contacts for row in contact["contact_constraint_rows"]
            ]
            result[side] = {
                "normal_force_N": stats(contact["normal_force_N"] for contact in contacts),
                "tangential_force_N": stats(
                    contact["tangential_force_magnitude_N"] for contact in contacts
                ),
                "relative_tangential_slip_speed_m_s": stats(
                    contact["relative_tangential_speed_m_s"] for contact in contacts
                ),
                "pyramidal_full_condim4_utilization": stats(
                    contact["cone_utilization"]["pyramidal_full_condim4_l1"]
                    for contact in contacts
                ),
                "elliptic_full_condim4_utilization": stats(
                    contact["cone_utilization"]["elliptic_full_condim4_l2"]
                    for contact in contacts
                ),
                "solver_constraint_rows_by_state": {
                    state: sum(row["state"] == state for row in constraint_rows)
                    for state in sorted({row["state"] for row in constraint_rows})
                },
                "solver_constraint_rows_nonzero_cost": sum(
                    bool(row["nonzero_cost_state"]) for row in constraint_rows
                ),
                "solver_contact_rows_total": len(constraint_rows),
            }
        return result

    qstart = float(samples[0]["bottle_world_position_m"][2])
    qend = float(samples[-1]["bottle_world_position_m"][2])
    return {
        "sample_count": len(samples),
        "ramp_samples": len(ramp),
        "hold_samples": len(hold),
        "bottle_vertical_displacement_sampled_m": qend - qstart,
        "maximum_bottle_vertical_displacement_m": max(
            float(row["bottle_world_position_m"][2]) for row in samples
        )
        - qstart,
        "hand_signed_world_vertical_support_N": {
            "ramp": stats(row["hand_signed_world_vertical_support_N"] for row in ramp),
            "hold": stats(row["hand_signed_world_vertical_support_N"] for row in hold),
        },
        "table_signed_world_vertical_support_N": {
            "ramp": stats(row["table_signed_world_vertical_support_N"] for row in ramp),
            "hold": stats(row["table_signed_world_vertical_support_N"] for row in hold),
        },
        "contact_fractions": {
            "bilateral_finger_contact": float(
                np.mean(
                    [
                        LEFT_PAD_BODY in row["hand_contact_bodies"]
                        and RIGHT_PAD_BODY in row["hand_contact_bodies"]
                        for row in samples
                    ]
                )
            ),
            "table_contact": float(np.mean([row["table_contact_count"] > 0 for row in samples])),
        },
        "per_finger_contact_statistics": {
            "ramp": stage_contact_statistics(ramp),
            "hold": stage_contact_statistics(hold),
        },
        "lift_actuator_tracking_error_m": stats(
            abs(row["actuators"]["lift"]["tracking_error_q_target_minus_actual"])
            for row in samples
        ),
        "lift_actuator_force_N": stats(
            abs(row["actuators"]["lift"]["actuator_force_N"]) for row in samples
        ),
        "actuator_saturated_sample_count": {
            name: sum(bool(row["actuators"][name]["saturated"]) for row in samples)
            for name in ("left", "right", "lift")
        },
        "solver_niter_per_step_max": max(
            (max(row["solver"]["solver_niter"], default=0) for row in samples), default=0
        ),
        "solver_niter_per_step_p95": float(
            np.quantile(
                [max(row["solver"]["solver_niter"], default=0) for row in samples], 0.95
            )
        ),
        "solver_gradient_max": max(
            (
                abs(stat["gradient"])
                for row in samples
                for stat in row["solver"]["solver_iteration_stats_initial_and_performed"]
            ),
            default=0.0,
        ),
        "maximum_penetration_m": max(
            (contact["penetration_m"] for row in samples for contact in row["contacts"]),
            default=0.0,
        ),
        "finite": all(
            math.isfinite(value)
            for row in samples
            for value in (
                row["hand_signed_world_vertical_support_N"],
                row["table_signed_world_vertical_support_N"],
                row["bottle_world_position_m"][2],
            )
        ),
    }


def model_identity_checks(pyramidal: mujoco.MjModel, elliptic: mujoco.MjModel) -> dict[str, Any]:
    if not mujoco.mj_isPyramidal(pyramidal) or mujoco.mj_isPyramidal(elliptic):
        raise RuntimeError("MuJoCo did not compile the requested pyramidal/elliptic cone pair")
    if mujoco.mj_versionString() != EXPECTED_MUJOCO_VERSION:
        raise RuntimeError(f"Expected MuJoCo {EXPECTED_MUJOCO_VERSION}, got {mujoco.mj_versionString()}")
    if (pyramidal.nq, pyramidal.nv, pyramidal.nu, pyramidal.na, pyramidal.ngeom, pyramidal.nbody) != (
        elliptic.nq,
        elliptic.nv,
        elliptic.nu,
        elliptic.na,
        elliptic.ngeom,
        elliptic.nbody,
    ):
        raise RuntimeError("Cone models differ in compiled topology")
    if pyramidal.na != 0:
        raise RuntimeError("This baseline has no stateful actuator activations; found na != 0")
    compared_arrays = (
        "qpos0",
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "jnt_range",
        "dof_armature",
        "dof_damping",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_gainprm",
        "actuator_biasprm",
    )
    mismatches = [
        name
        for name in compared_arrays
        if not np.array_equal(getattr(pyramidal, name), getattr(elliptic, name))
    ]
    if mismatches:
        raise RuntimeError(f"Compiled model changed outside cone selection: {mismatches}")
    for name in ("timestep", "integrator", "solver", "iterations", "tolerance", "impratio"):
        if getattr(pyramidal.opt, name) != getattr(elliptic.opt, name):
            raise RuntimeError(f"Model option {name} differs between A/B")
    if not np.array_equal(pyramidal.opt.gravity, elliptic.opt.gravity):
        raise RuntimeError("Gravity differs between A/B")
    return {
        "only_allowed_compiled_option_difference": "cone",
        "pyramidal_cone_enum": int(pyramidal.opt.cone),
        "elliptic_cone_enum": int(elliptic.opt.cone),
        "compiled_topology": {
            "nq": int(pyramidal.nq),
            "nv": int(pyramidal.nv),
            "nu": int(pyramidal.nu),
            "na": int(pyramidal.na),
            "ngeom": int(pyramidal.ngeom),
            "nbody": int(pyramidal.nbody),
        },
        "bitwise_equal_model_arrays": list(compared_arrays),
        "same_gravity_timestep_integrator_solver_iterations_tolerance_impratio": True,
    }


def run_configuration(
    name: str,
    model: mujoco.MjModel,
    source_row: dict[str, Any],
    output: Path,
    xml_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    xml_path = output / f"model_{name}.xml"
    trace_path = output / f"steps_{name}.jsonl"
    xml_path.write_text(xml_text, encoding="utf-8")
    data, restore_info = restore_state(model, source_row)
    dt = float(model.opt.timestep)
    total_steps = round((RAMP_DURATION_S + HOLD_DURATION_S) / dt)
    lift_actuator = actuator_id(model, "bounded_vertical_carriage_servo")
    left_actuator = actuator_id(model, "left_pad_bounded_servo")
    right_actuator = actuator_id(model, "right_pad_bounded_servo")
    type_names = enum_names(mujoco.mjtConstraint)
    state_names = enum_names(mujoco.mjtConstraintState)
    samples = []
    with trace_path.open("w", encoding="utf-8") as stream:
        for step in range(total_steps):
            elapsed = step * dt
            s = min(1.0, max(0.0, elapsed / RAMP_DURATION_S))
            smooth = s * s * (3.0 - 2.0 * s)
            lift_target = LIFT_BASE_TARGET_M + DIAGNOSTIC_LIFT_M * smooth
            data.ctrl[left_actuator] = PAD_TARGET_M
            data.ctrl[right_actuator] = PAD_TARGET_M
            data.ctrl[lift_actuator] = lift_target
            mujoco.mj_forward(model, data)
            sample = sample_step(model, data, elapsed, lift_target, type_names, state_names)
            sample["step"] = step
            samples.append(sample)
            stream.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
            mujoco.mj_step(model, data)
    summary = {
        "configuration": name,
        "cone": "pyramidal" if mujoco.mj_isPyramidal(model) else "elliptic",
        "model_xml_sha256": sha256_file(xml_path),
        "step_trace_sha256": sha256_file(trace_path),
        "restored_state": restore_info,
        "measurements": summarize(samples),
    }
    return summary, restore_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Immutable run06 evidence directory")
    parser.add_argument("--output", type=Path, required=True, help="New, empty evidence directory")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Compile both cone models and validate state/contact instrumentation without stepping",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output = args.output.resolve()
    source_xml = source_dir / "ideal_parallel_gripper.xml"
    source_trace = source_dir / "physics_trace.jsonl"
    source_result = source_dir / "result.json"
    if not all(path.is_file() for path in (source_xml, source_trace, source_result)):
        raise SystemExit(f"Missing run06 evidence in {source_dir}")
    if sha256_file(source_xml) != EXPECTED_SOURCE_XML_SHA256:
        raise SystemExit("run06 source XML SHA-256 does not match the audited immutable baseline")
    if sha256_file(source_trace) != EXPECTED_SOURCE_TRACE_SHA256:
        raise SystemExit("run06 source trace SHA-256 does not match the audited immutable baseline")
    result = json.loads(source_result.read_text(encoding="utf-8"))
    if result["runtime_identity"]["robot_sim_head"] != EXPECTED_SOURCE_RUN_SHA:
        raise SystemExit("run06 result records a different RobotSim source SHA")
    if result["runtime_identity"]["mujoco_python_version"] != EXPECTED_MUJOCO_VERSION:
        raise SystemExit("run06 result records a different MuJoCo version")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing evidence directory: {output}")

    rows = read_trace(source_trace)
    hold_rows = [row for row in rows if row["phase"] == SOURCE_PHASE]
    if not hold_rows:
        raise SystemExit(f"No {SOURCE_PHASE} rows in source trace")
    source_row = max(hold_rows, key=lambda row: int(row["step"]))
    if source_row["bottle_qpos_write_count"] != 0 or source_row["mocap_carry"]:
        raise SystemExit("Selected run06 hold snapshot violates the no-write/no-mocap condition")
    if source_row["equalities"] != 0 or source_row["joints_actuators"]["left"]["saturated"] or source_row[
        "joints_actuators"
    ]["right"]["saturated"]:
        raise SystemExit("Selected hold snapshot is not a clean bilateral hold state")

    source_xml_text = source_xml.read_text(encoding="utf-8")
    pyramid_xml = inject_cone(source_xml_text, "pyramidal")
    elliptic_xml = inject_cone(source_xml_text, "elliptic")
    normalize = lambda value: re.sub(r'\s+cone="(?:pyramidal|elliptic)"', "", value)
    if normalize(pyramid_xml) != source_xml_text or normalize(elliptic_xml) != source_xml_text:
        raise RuntimeError("Generated XML differs from the source by more than the cone attribute")

    model_p = mujoco.MjModel.from_xml_string(pyramid_xml)
    model_e = mujoco.MjModel.from_xml_string(elliptic_xml)
    model_check = model_identity_checks(model_p, model_e)
    if args.preflight_only:
        type_names = enum_names(mujoco.mjtConstraint)
        state_names = enum_names(mujoco.mjtConstraintState)
        preflight = {}
        for name, model in (("pyramidal", model_p), ("elliptic", model_e)):
            data, restore_info = restore_state(model, source_row)
            sample = sample_step(
                model,
                data,
                0.0,
                float(source_row["command"]["lift_target_q_m"]),
                type_names,
                state_names,
            )
            pairs = {tuple(row["pair"]) for row in sample["contacts"]}
            hand_bodies = {
                row.get("other_body")
                for row in sample["contacts"]
                if row.get("support_class") == "hand_pad"
            }
            if hand_bodies != {LEFT_PAD_BODY, RIGHT_PAD_BODY}:
                raise RuntimeError(f"Restored {name} state does not have both pad contacts: {hand_bodies}")
            if sample["table_contact_count"] < 1 or sample["bottle_contact_count"] < 3:
                raise RuntimeError(f"Restored {name} state lost expected table/pad contact")
            preflight[name] = {
                "restored_state": {
                    "source_trace_step": restore_info["source_trace_step"],
                    "source_trace_time_s": restore_info["source_trace_time_s"],
                    "state_sha256": restore_info["state_sha256"],
                    "rebuild_validation": restore_info["rebuild_validation"],
                },
                "time_s": sample["time_s"],
                "contact_count_all": sample["contact_count_all"],
                "bottle_contact_count": sample["bottle_contact_count"],
                "contact_pairs": sorted([list(pair) for pair in pairs]),
                "solver": sample["solver"],
                "contact_force_check": [
                    {
                        "pair": row["pair"],
                        "support_class": row["support_class"],
                        "friction_coefficients_compiled": row.get("friction_coefficients_compiled"),
                        "condim": row.get("condim"),
                        "solref": row.get("solref"),
                        "solimp": row.get("solimp"),
                        "normal_force_N": row.get("normal_force_N"),
                        "tangential_force_magnitude_N": row.get("tangential_force_magnitude_N"),
                        "relative_tangential_speed_m_s": row.get("relative_tangential_speed_m_s"),
                        "constraint_type_state_counts": {
                            f"{constraint['type']}:{constraint['state']}": sum(
                                candidate["type"] == constraint["type"]
                                and candidate["state"] == constraint["state"]
                                for candidate in row["contact_constraint_rows"]
                            )
                            for constraint in row["contact_constraint_rows"]
                        },
                    }
                    for row in sample["contacts"]
                ],
            }
        if preflight["pyramidal"]["restored_state"]["state_sha256"] != preflight["elliptic"][
            "restored_state"
        ]["state_sha256"]:
            raise RuntimeError("Zero-step A/B restored states are not identical")
        contact_signature = lambda case: [
            (
                tuple(row["pair"]),
                tuple(row["friction_coefficients_compiled"] or []),
                row["condim"],
                tuple(row["solref"] or []),
                tuple(row["solimp"] or []),
            )
            for row in case["contact_force_check"]
        ]
        if contact_signature(preflight["pyramidal"]) != contact_signature(preflight["elliptic"]):
            raise RuntimeError("Compiled per-contact friction/condim/solref/solimp differs in zero-step state")
        print(
            json.dumps(
                {
                    "preflight_only": True,
                    "no_mj_step_called": True,
                    "source_run06_xml_unchanged": sha256_file(source_xml) == EXPECTED_SOURCE_XML_SHA256,
                    "model_identity_checks": model_check,
                    "configurations": preflight,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    output.mkdir(parents=True, exist_ok=False)

    try:
        result_p, restore_p = run_configuration("pyramidal", model_p, source_row, output, pyramid_xml)
        result_e, restore_e = run_configuration("elliptic", model_e, source_row, output, elliptic_xml)
        if restore_p["state_sha256"] != restore_e["state_sha256"]:
            raise RuntimeError("A/B restorations are not byte-identical")

        result_document = {
            "classification": "SIMULATION_ONLY_IDEAL_PARALLEL_GRIPPER_FRICTION_CONE_DIAGNOSTIC",
            "status": "DIAGNOSTIC_COMPLETE_NOT_GRASP_ACCEPTANCE",
            "source_run06": {
                "directory": str(source_dir),
                "source_xml_sha256": EXPECTED_SOURCE_XML_SHA256,
                "source_trace_sha256": EXPECTED_SOURCE_TRACE_SHA256,
                "source_result_sha256": sha256_file(source_result),
                "robot_sim_sha": EXPECTED_SOURCE_RUN_SHA,
                "mujoco_version": mujoco.mj_versionString(),
                "mujoco_python_module": str(Path(mujoco.__file__).resolve()),
                "native_libraries": result["runtime_identity"].get("native_libraries", []),
                "source_scene_identity": result["scene"],
                "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            "experiment": {
                "only_model_change": "option cone: pyramidal vs elliptic",
                "state_source_phase": SOURCE_PHASE,
                "state_source_step": int(source_row["step"]),
                "state_source_time_s": float(source_row["time_s"]),
                "controlled_motion": {
                    "type": "vertical carriage position reference; same for A and B",
                    "pad_target_q_m": PAD_TARGET_M,
                    "lift_start_target_q_m": LIFT_BASE_TARGET_M,
                    "vertical_reference_displacement_m": DIAGNOSTIC_LIFT_M,
                    "smoothstep_duration_s": RAMP_DURATION_S,
                    "post-ramp_hold_s": HOLD_DURATION_S,
                    "total_steps_per_configuration": round(
                        (RAMP_DURATION_S + HOLD_DURATION_S) / float(model_p.opt.timestep)
                    ),
                    "object_qpos_writes_after_restore": 0,
                    "weld_mocap_or_artificial_object_force": False,
                },
                "model_identity_checks": model_check,
                "restored_state_identical": restore_p["state_sha256"] == restore_e["state_sha256"],
                "nominal_model_parameters_unchanged": True,
            },
            "runs": {"pyramidal": result_p, "elliptic": result_e},
            "artifacts": {
                "steps_pyramidal": "steps_pyramidal.jsonl",
                "steps_elliptic": "steps_elliptic.jsonl",
                "model_pyramidal": "model_pyramidal.xml",
                "model_elliptic": "model_elliptic.xml",
                "restored_state": "restored_state.json",
            },
        }
        write_json(output / "restored_state.json", restore_p)
        write_json(output / "result.json", result_document)
        print(json.dumps(result_document, indent=2, sort_keys=True))
    except Exception:
        # Preserve partial raw evidence for diagnosis; never silently delete it.
        raise


if __name__ == "__main__":
    main()
