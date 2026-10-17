#!/usr/bin/env python3
"""Compare active MuJoCo hull contacts with raw X2 STL triangle surfaces."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/x2-omnipicker-closed-loop-dynamics-20261017-run01")
SOURCE_MESH_DIR = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0/meshes")
TRIANGLE_AUDITOR = Path("/home/lzy18001500226/robotsim-issue46-omnipicker-linkage-root-cause-20261014/scripts/research/issue46_omnipicker_linkage_root_cause.py")
PREFIX = "x2cand_candidate_"
ROOT_JOINT = PREFIX + "right_claw_joint"
FOLLOWER_JOINT = PREFIX + "R_hand_wide1_joint"
PASSIVE = {
    "narrow": (PREFIX + "R_hand_narrow2_passive_hinge", PREFIX + "R_hand_narrow3_passive_hinge", PREFIX + "R_hand_narrow_loop_passive_hinge"),
    "wide": (PREFIX + "R_hand_wide2_passive_hinge", PREFIX + "R_hand_wide3_passive_hinge", PREFIX + "R_hand_wide_loop_passive_hinge"),
}
PAIRS = (
    ("R_hand_narrow1_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow2_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_wide1_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide2_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide3_Link", "R_hand_wide_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_wide3_Link"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oid(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise KeyError(name)
    return value


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or str(body_id)


def mesh_path(body: str) -> Path:
    if body == "R_omnipicker_base_link":
        return SOURCE_MESH_DIR / "base_link.stl"
    stem = body.removeprefix("R_hand_")
    return SOURCE_MESH_DIR / f"{stem}.stl"


def world_triangles(model: mujoco.MjModel, data: mujoco.MjData, body: str, triangles_fn) -> tuple[np.ndarray, str]:
    path = mesh_path(body)
    triangles = triangles_fn(path)
    bid = oid(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + body)
    rotation = data.xmat[bid].reshape(3, 3)
    world = triangles @ rotation.T + data.xpos[bid]
    return world, sha256(path)


def geom_pair_distance(model: mujoco.MjModel, data: mujoco.MjData, body1: str, body2: str) -> dict:
    ids1 = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == oid(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + body1)
            and (int(model.geom_contype[i]) != 0 or int(model.geom_conaffinity[i]) != 0)]
    ids2 = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == oid(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + body2)
            and (int(model.geom_contype[i]) != 0 or int(model.geom_conaffinity[i]) != 0)]
    values = []
    segment = np.zeros(6)
    for a in ids1:
        for b in ids2:
            distance = float(mujoco.mj_geomDistance(model, data, a, b, 1.0, segment))
            values.append({"geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a),
                           "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b),
                           "signed_distance_mm": distance * 1000.0})
    return {"pairwise": values, "minimum_signed_distance_mm": min((x["signed_distance_mm"] for x in values), default=None)}


def main() -> None:
    spec = importlib.util.spec_from_file_location("issue46_triangle_auditor", TRIANGLE_AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load retained triangle auditor: {TRIANGLE_AUDITOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    models = {
        "tree_control": mujoco.MjModel.from_xml_path(str(ROOT / "source_tree_control.xml")),
        "closed_loop_candidate": mujoco.MjModel.from_xml_path(str(ROOT / "passive_loop_candidate.xml")),
    }
    static = json.loads((ROOT / "closed_loop_static_scan.json").read_text(encoding="utf-8"))["rows"]
    states = []
    for q in (0.0, -0.5, -1.0):
        static_row = min(static, key=lambda row: abs(float(row["driver_q_rad"]) - q))
        for variant, model in models.items():
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            jid = oid(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)
            fid = oid(model, mujoco.mjtObj.mjOBJ_JOINT, FOLLOWER_JOINT)
            data.qpos[int(model.jnt_qposadr[jid])] = q
            data.qpos[int(model.jnt_qposadr[fid])] = -q
            if variant == "closed_loop_candidate":
                for side in ("narrow", "wide"):
                    for joint, value in static_row["sides"][side]["passive_joint_rad"].items():
                        pid = oid(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                        data.qpos[int(model.jnt_qposadr[pid])] = float(value)
            mujoco.mj_forward(model, data)
            active = []
            for ci in range(data.ncon):
                c = data.contact[ci]
                b1 = body_name(model, int(model.geom_bodyid[int(c.geom1)]))
                b2 = body_name(model, int(model.geom_bodyid[int(c.geom2)]))
                active.append({"body1": b1, "body2": b2, "distance_mm": float(c.dist * 1000.0)})
            for body1, body2 in PAIRS:
                a, sha_a = world_triangles(model, data, body1, module.read_binary_stl)
                b, sha_b = world_triangles(model, data, body2, module.read_binary_stl)
                surface = module.triangle_mesh_intersections(a, b, pair_limit=200000)
                active_rows = [c for c in active if {c["body1"].removeprefix(PREFIX), c["body2"].removeprefix(PREFIX)} == {body1, body2}]
                states.append({
                    "variant": variant, "driver_q_rad": q,
                    "body1": body1, "body2": body2,
                    "mesh_sha256": {body1: sha_a, body2: sha_b},
                    "compiled_convex_distance": geom_pair_distance(model, data, body1, body2),
                    "active_solver_contacts": active_rows,
                    "raw_triangle_surface_test": surface,
                })
    result = {
        "classification": "offline raw triangle surface cross-check; does not alter models or advance physics",
        "mujoco_version": mujoco.__version__,
        "triangle_auditor_path": str(TRIANGLE_AUDITOR),
        "triangle_auditor_sha256": sha256(TRIANGLE_AUDITOR),
        "sample_q_rad": [0.0, -0.5, -1.0],
        "surface_intersection_limitations": "Triangle surface intersections do not detect full containment; compiled MuJoCo distances are convex-hull proxy values.",
        "results": states,
    }
    out = ROOT / "contact_surface_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": len(states), "status_counts": {
        status: sum(1 for row in states if row["raw_triangle_surface_test"]["status"] == status)
        for status in sorted({row["raw_triangle_surface_test"]["status"] for row in states})
    }}, indent=2))


if __name__ == "__main__":
    main()
