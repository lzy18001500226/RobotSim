"""Table-only task geometry and initialization grid; no live object feedback."""
from __future__ import annotations
import math
import numpy as np

FRAME = "native_scene_world"
STEP_M = .08
CENTER_INSET_M = .055
OBSERVATION_STANDOFF_M = .50

def table_bounds(model, data, name="raicom_table_table"):
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError("required source table missing")
    vertices = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        vertices.extend(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    if not vertices:
        raise ValueError("source table mesh bounds unavailable")
    v = np.asarray(vertices)
    return {"min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist()}

def observation_goal(bounds):
    lo, hi = np.asarray(bounds["min"]), np.asarray(bounds["max"])
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(np.r_[lo, hi]).all() or np.any(hi <= lo):
        raise ValueError("finite ordered world table bounds required")
    if hi[1]-lo[1] <= hi[0]-lo[0]:
        raise ValueError("this reviewed route requires world-Y long edge")
    return {"xy_m": [float(lo[0]-OBSERVATION_STANDOFF_M), float((lo[1]+hi[1])/2)],
            "yaw_rad": 0., "frame": FRAME, "source": "TABLE_GEOMETRY_ONLY",
            "standoff_m": OBSERVATION_STANDOFF_M,
            "qualification": "STATIC_RGBD_ALL11; ACTUAL_ARRIVAL_MUST_REOBSERVE"}

def initial_positions(bounds, original_object_xy):
    """Evaluator/setup only. Never pass these object coordinates to live control."""
    original = np.asarray(original_object_xy, dtype=float)
    if original.shape != (2,) or not np.isfinite(original).all():
        raise ValueError("finite initialization object XY required")
    observation_goal(bounds)
    lo, hi = np.asarray(bounds["min"]), np.asarray(bounds["max"])
    if not lo[0]+CENTER_INSET_M <= original[0] <= hi[0]-CENTER_INSET_M:
        raise ValueError("initial object centerline outside supported inset")
    start, end = lo[1]+CENTER_INSET_M, hi[1]-CENTER_INSET_M
    if start >= end:
        raise ValueError("no supported tabletop interval")
    ys = [start+k*STEP_M for k in range(math.floor((end-start)/STEP_M)+1)]
    if end-ys[-1] > 1e-9:
        ys.append(end)
    return [{"id": i, "initial_xy_m": [float(original[0]), float(y)],
             "offset_xy_m": [0., float(y-original[1])],
             "initialization_only_not_control_target": True} for i, y in enumerate(ys)]
