#!/usr/bin/env python3
"""Audit one compiled DFQ candidate using fresh, zero-step MuJoCo contacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from g1_dfq_static_wrench import ContactWrench, solve_static_wrench  # noqa: E402


BOTTLE_GEOMS = {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_joint(model, data, name: str, value: float) -> None:
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if jid < 0:
        raise RuntimeError(f"missing candidate joint: {name}")
    data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def load_candidate_state(model, candidate: dict, base: dict, urdf_path: Path) -> mujoco.MjData:
    import g1_inspire_hand as hand

    hand.URDF = urdf_path
    _, _, mimics, _ = hand.parse_urdf()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for name, value in base["joint_state_qpos_rad"].items():
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            set_joint(model, data, name, float(value))
    for name, record in candidate["joint_configuration"].items():
        set_joint(model, data, name, float(record["q_rad"]))

    free_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free"))
    if free_id < 0:
        raise RuntimeError("compiled model has no bottle_free joint")
    qadr = int(model.jnt_qposadr[free_id])
    data.qpos[qadr:qadr + 7] = np.r_[
        base["bottle_root_position_world_m"], base["bottle_root_quaternion_wxyz"]
    ]
    for _ in range(len(mimics) + 1):
        for follower, relation in mimics.items():
            parent_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, relation["parent"]))
            follower_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower))
            parent_q = float(data.qpos[int(model.jnt_qposadr[parent_id])])
            set_joint(
                model, data, follower,
                float(relation["multiplier"]) * parent_q + float(relation["offset"]),
            )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def active_bottle_hand_contacts(model, data) -> tuple[list[ContactWrench], list[dict]]:
    contacts: list[ContactWrench] = []
    rows: list[dict] = []
    bottle_geom_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in BOTTLE_GEOMS
    }
    bottle_geom_ids.discard(-1)
    for index in range(data.ncon):
        con = data.contact[index]
        geom1, geom2 = int(con.geom1), int(con.geom2)
        if (geom1 in bottle_geom_ids) == (geom2 in bottle_geom_ids):
            continue
        bottle_geom, hand_geom = (geom1, geom2) if geom1 in bottle_geom_ids else (geom2, geom1)
        hand_body = int(model.geom_bodyid[hand_geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, hand_body) or ""
        if not body_name.startswith("R_") or float(con.dist) > float(con.includemargin):
            continue
        sign = 1.0 if bottle_geom == geom2 else -1.0
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        bottle_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom) or ""
        hand_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom) or ""
        contact_name = f"{body_name}:{bottle_name}"
        contacts.append(ContactWrench(
            position_world_m=np.asarray(con.pos, dtype=float).copy(),
            frame_world_rows=np.asarray(con.frame, dtype=float).reshape(3, 3).copy(),
            friction=np.asarray(con.friction, dtype=float).copy(),
            condim=int(con.dim),
            force_on_bottle_sign=sign,
            name=contact_name,
        ))
        rows.append({
            "contact_index": index,
            "geom1": geom1_name,
            "geom2": geom2_name,
            "hand_body": body_name,
            "hand_geom": hand_name,
            "bottle_geom": bottle_name,
            "distance_m": float(con.dist),
            "active_by_compiled_margin": True,
            "condim": int(con.dim),
            "friction": np.asarray(con.friction, dtype=float).tolist(),
            "position_world_m": np.asarray(con.pos, dtype=float).tolist(),
            "frame_world_rows": np.asarray(con.frame, dtype=float).reshape(3, 3).tolist(),
            "force_on_bottle_sign": sign,
        })
    return contacts, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    default_ros = os.environ.get("ROBOTSIM_UNITREE_ROS_DIR")
    parser.add_argument("--urdf", type=Path, default=(
        Path(default_ros) / "robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
        if default_ros else None
    ))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.urdf is None:
        parser.error("set ROBOTSIM_UNITREE_ROS_DIR or pass --urdf")

    repo = PROJECT_ROOT
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    base = json.loads(args.base_state.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = load_candidate_state(model, candidate, base, args.urdf)
    free_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free"))
    bottle_body = int(model.jnt_bodyid[free_id])
    com = np.asarray(data.subtree_com[bottle_body], dtype=float)
    mass = float(model.body_subtreemass[bottle_body])
    target = np.r_[-mass * np.asarray(model.opt.gravity, dtype=float), np.zeros(3)]
    contacts, contact_rows = active_bottle_hand_contacts(model, data)
    full = solve_static_wrench(contacts, com, target, include_moments=True)
    force_only = solve_static_wrench(contacts, com, target, include_moments=False)
    if force_only["feasible"]:
        ray_matrix = np.asarray(force_only["contact_rays_world_about_com"], dtype=float)
        coeffs = np.asarray(force_only["ray_coefficients_n"], dtype=float)
        full["force_only_allocation_moment_residual_nm"] = (ray_matrix[3:, :] @ coeffs).tolist()
    try:
        robotsim_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        robotsim_dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
    except Exception:
        robotsim_head, robotsim_dirty = None, None
    payload = {
        "candidate": candidate.get("candidate", candidate.get("run_id")),
        "candidate_run_id": candidate.get("run_id"),
        "physics_steps": 0,
        "mj_step_calls": 0,
        "mj_forward_calls": 1,
        "robotsim_head": robotsim_head,
        "robotsim_dirty": robotsim_dirty,
        "candidate_sha256": sha256(args.candidate),
        "compiled_model_sha256": sha256(args.model),
        "base_state_sha256": sha256(args.base_state),
        "mujoco_version": mujoco.__version__,
        "cone": "pyramidal" if mujoco.mj_isPyramidal(model) else "elliptic",
        "contact_model": {
            "bottle_mass_kg": mass,
            "bottle_com_world_m": com.tolist(),
            "gravity_world_m_s2": np.asarray(model.opt.gravity, dtype=float).tolist(),
            "required_hand_wrench_about_com": target.tolist(),
            "contact_count": len(contacts),
            "contacts": contact_rows,
            "contact_basis": "MuJoCo pyramidal cone: two nonnegative rays per friction dimension; condim=4 gives six rays including torsional friction",
        },
        "full_six_dimensional_equilibrium": full,
        "force_only_diagnostic": force_only,
        "geometry_gate": candidate.get("static_gates", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate": payload["candidate"],
        "active_contact_count": len(contacts),
        "full_6d_feasible": full["feasible"],
        "force_only_feasible": force_only["feasible"],
        "physics_steps": 0,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
