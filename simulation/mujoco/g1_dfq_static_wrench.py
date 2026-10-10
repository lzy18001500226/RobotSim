"""Static contact-wrench feasibility using MuJoCo pyramidal contact cones."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class ContactWrench:
    """One active contact expressed in world coordinates."""

    position_world_m: np.ndarray
    frame_world_rows: np.ndarray
    friction: np.ndarray
    condim: int
    force_on_bottle_sign: float
    name: str = ""
    hand_body_id: int | None = None


def pyramidal_wrench_rays(contact: ContactWrench, bottle_com_world_m: np.ndarray) -> np.ndarray:
    """Return world 6D wrench rays about the bottle COM for this contact.

    MuJoCo's pyramidal cone uses two nonnegative rays per friction dimension.
    For condim=4 these are normal +/- each slide direction and normal +/- spin.
    The frame stores its axes in rows. `force_on_bottle_sign` is +1 when the
    bottle is geom2 in MuJoCo's geom1-to-geom2 convention, otherwise -1.
    """
    if contact.condim < 1 or contact.condim > 4:
        raise ValueError(f"unsupported contact condim {contact.condim}")
    frame = np.asarray(contact.frame_world_rows, dtype=float).reshape(3, 3)
    point = np.asarray(contact.position_world_m, dtype=float).reshape(3)
    com = np.asarray(bottle_com_world_m, dtype=float).reshape(3)
    friction = np.asarray(contact.friction, dtype=float).reshape(-1)
    expected = contact.condim - 1
    if friction.size < expected:
        raise ValueError("contact friction vector is shorter than condim requires")
    if not np.isclose(abs(contact.force_on_bottle_sign), 1.0):
        raise ValueError("force_on_bottle_sign must be -1 or +1")

    rays: list[np.ndarray] = []
    for friction_dim in range(1, contact.condim):
        mu = float(friction[friction_dim - 1])
        for direction in (1.0, -1.0):
            local_force = np.zeros(3)
            local_torque = np.zeros(3)
            if friction_dim <= 2:
                local_force[friction_dim] = direction * mu
            else:
                local_torque[0] = direction * mu
            local_force[0] = 1.0

            force_world = contact.force_on_bottle_sign * (frame.T @ local_force)
            torque_world = contact.force_on_bottle_sign * (frame.T @ local_torque)
            moment_world = np.cross(point - com, force_world) + torque_world
            rays.append(np.r_[force_world, moment_world])
    return np.column_stack(rays) if rays else np.zeros((6, 0))


def solve_static_wrench(
    contacts: list[ContactWrench],
    bottle_com_world_m: np.ndarray,
    target_wrench_world: np.ndarray,
    *,
    include_moments: bool = True,
    effort_map_nm_per_ray: np.ndarray | None = None,
    effort_limit_nm: float | None = None,
) -> dict:
    """Solve unilateral nonnegative ray forces for the requested bottle wrench."""
    if not contacts:
        return {"feasible": False, "reason": "no_active_contacts", "contact_count": 0}

    blocks = [pyramidal_wrench_rays(contact, bottle_com_world_m) for contact in contacts]
    matrix = np.column_stack(blocks)
    target = np.asarray(target_wrench_world, dtype=float).reshape(6)
    rows = slice(0, 6) if include_moments else slice(0, 3)
    objective = np.ones(matrix.shape[1])
    a_ub = b_ub = None
    if effort_limit_nm is not None:
        if effort_limit_nm <= 0 or effort_map_nm_per_ray is None:
            raise ValueError("positive effort limit requires an effort map")
        effort = np.asarray(effort_map_nm_per_ray, dtype=float)
        if effort.ndim != 2 or effort.shape[1] != matrix.shape[1]:
            raise ValueError("effort map must have one column per wrench ray")
        a_ub = np.vstack((effort, -effort))
        b_ub = np.full(a_ub.shape[0], float(effort_limit_nm))

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=matrix[rows],
        b_eq=target[rows],
        bounds=(0.0, None),
        method="highs",
    )
    output = {
        "feasible": bool(result.success),
        "solver_status": str(result.message),
        "contact_count": len(contacts),
        "ray_count": int(matrix.shape[1]),
        "equilibrium_dimension": 6 if include_moments else 3,
        "target_wrench_world": target.tolist(),
        "contact_names": [contact.name for contact in contacts],
        "contact_rays_world_about_com": matrix.tolist(),
    }
    if result.success:
        residual = matrix @ result.x - target
        output.update({
            "ray_coefficients_n": result.x.tolist(),
            "equilibrium_residual_world": residual.tolist(),
            "normal_load_proxy_n": float(np.sum(result.x)),
        })
    return output


def candidate_rollout_allocation(
    candidate: dict,
    *,
    bottle_weight_n: float,
    effort_cap_nm: float,
    equilibrium_tolerance: float = 1e-6,
) -> dict:
    """Convert a passing search candidate into a rollout-specific LP record.

    A condim=4 pyramidal contact has six rays. Each ray carries one unit of
    local normal force, so its six coefficients sum to the contact normal
    component. This is a static target only; dynamic support is measured from
    current contacts during the rollout.
    """
    if candidate.get("static_candidate_pass") is not True:
        raise ValueError("geometry/wrench candidate did not pass its static gates")
    if candidate.get("robust_perturbations_pass") is not True:
        raise ValueError("candidate contact perturbation checks did not all pass")

    equilibrium = candidate.get("effort_bounded_6d_equilibrium", {})
    if equilibrium.get("feasible") is not True or equilibrium.get("equilibrium_dimension") != 6:
        raise ValueError("candidate has no feasible full 6D static wrench solution")
    names = list(equilibrium.get("contact_names", []))
    coefficients = np.asarray(equilibrium.get("ray_coefficients_n", []), dtype=float)
    if not names or coefficients.shape != (len(names) * 6,):
        raise ValueError("condim=4 candidate solution must contain six rays per contact")
    if not np.all(np.isfinite(coefficients)) or np.min(coefficients) < -1e-10:
        raise ValueError("candidate has invalid or negative contact-ray coefficients")

    contact_by_name = {
        f"{row.get('body')}:{row.get('bottle_geom')}": row
        for row in candidate.get("active_right_hand_bottle_contacts", [])
    }
    per_contact = []
    for index, name in enumerate(names):
        source = contact_by_name.get(name)
        if source is None:
            raise ValueError(f"static solution contact does not match a geometry row: {name}")
        normal_force = float(np.sum(coefficients[index * 6:(index + 1) * 6]))
        per_contact.append({
            "contact": name,
            "digit": str(source["digit"]),
            "bottle_geom": str(source["bottle_geom"]),
            "normal_force_n": normal_force,
            "normal_force_definition": "sum of six nonnegative condim=4 cone-ray coefficients",
        })

    target = np.asarray(equilibrium.get("target_wrench_world", []), dtype=float)
    residual = np.asarray(equilibrium.get("equilibrium_residual_world", []), dtype=float)
    if target.shape != (6,) or residual.shape != (6,):
        raise ValueError("candidate wrench target and residual must each have six components")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(residual)):
        raise ValueError("candidate wrench contains non-finite values")
    if np.linalg.norm(residual, ord=np.inf) > equilibrium_tolerance:
        raise ValueError("candidate static wrench residual exceeds the accepted tolerance")
    if not math.isclose(float(target[2]), bottle_weight_n, rel_tol=0.0, abs_tol=0.005):
        raise ValueError("candidate wrench target does not match bottle weight")
    if np.linalg.norm(np.delete(target, 2), ord=np.inf) > equilibrium_tolerance:
        raise ValueError("candidate static target includes unintended force or moment components")

    digit_normal = {}
    for item in per_contact:
        digit_normal[item["digit"]] = digit_normal.get(item["digit"], 0.0) + item["normal_force_n"]
    opposing = sum(digit_normal.get(name, 0.0) for name in ("index", "middle", "ring", "pinky"))
    if digit_normal.get("thumb", 0.0) <= 0.0 or opposing <= 0.0:
        raise ValueError("candidate static allocation lacks thumb or opposing-digit force")

    raw_effort = equilibrium.get("estimated_driver_effort_nm", {})
    driver_efforts = {}
    for joint_name, value in raw_effort.items():
        prefix, suffix = "R_", "_joint"
        if not joint_name.startswith(prefix) or not joint_name.endswith(suffix):
            raise ValueError(f"unexpected static effort joint name: {joint_name}")
        channel = joint_name[len(prefix):-len(suffix)]
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"non-finite static effort for {joint_name}")
        driver_efforts[channel] = numeric
    expected_channels = {
        "pinky_proximal", "ring_proximal", "middle_proximal",
        "index_proximal", "thumb_proximal_pitch", "thumb_proximal_yaw",
    }
    if set(driver_efforts) != expected_channels:
        raise ValueError("candidate static effort does not cover all six independent channels")
    peak_effort = max(abs(value) for value in driver_efforts.values())
    if not math.isfinite(effort_cap_nm) or peak_effort > effort_cap_nm + 1e-9:
        raise ValueError("candidate static effort exceeds the unchanged simulation cap")

    return {
        "schema": "robotsim.g1_dfq.candidate_static_wrench_allocation.v1",
        "feasible": True,
        "equilibrium_dimension": 6,
        "target_wrench_world": target.tolist(),
        "equilibrium_residual_world": residual.tolist(),
        "bottle_weight_n": float(bottle_weight_n),
        "per_contact": per_contact,
        "per_digit_normal_force_n": digit_normal,
        "total_normal_force_n": float(sum(item["normal_force_n"] for item in per_contact)),
        "driver_efforts_nm": driver_efforts,
        "maximum_absolute_driver_effort_nm": peak_effort,
        "effort_cap_nm": float(effort_cap_nm),
        "contact_names": names,
        "robust_contact_perturbations_pass": True,
    }
