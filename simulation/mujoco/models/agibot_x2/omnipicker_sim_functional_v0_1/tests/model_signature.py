"""Stable compiled-MJCF signature used for source-to-portable static comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _name(model, object_type, object_id):
    import mujoco

    return mujoco.mj_id2name(model, object_type, object_id) or ""


def _float_array(value):
    return np.asarray(value, dtype="<f8").tolist()


def _array_digest(value, dtype):
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compiled_model_signature(model, xml_path: Path) -> dict:
    """Hash model structure, source-mesh arrays, contacts, and physics config."""
    import mujoco

    joints = []
    for jid in range(model.njnt):
        joints.append({
            "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
            "type": int(model.jnt_type[jid]),
            "bodyid": int(model.jnt_bodyid[jid]),
            "pos": _float_array(model.jnt_pos[jid]),
            "axis": _float_array(model.jnt_axis[jid]),
            "limited": int(model.jnt_limited[jid]),
            "range": _float_array(model.jnt_range[jid]),
            "qposadr": int(model.jnt_qposadr[jid]),
            "dofadr": int(model.jnt_dofadr[jid]),
        })

    bodies = []
    for bid in range(model.nbody):
        bodies.append({
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
            "parentid": int(model.body_parentid[bid]),
            "pos": _float_array(model.body_pos[bid]),
            "quat": _float_array(model.body_quat[bid]),
            "mass": float(model.body_mass[bid]),
            "inertia": _float_array(model.body_inertia[bid]),
            "ipos": _float_array(model.body_ipos[bid]),
            "iquat": _float_array(model.body_iquat[bid]),
        })

    geoms = []
    for gid in range(model.ngeom):
        geoms.append({
            "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
            "type": int(model.geom_type[gid]),
            "bodyid": int(model.geom_bodyid[gid]),
            "dataid": int(model.geom_dataid[gid]),
            "pos": _float_array(model.geom_pos[gid]),
            "quat": _float_array(model.geom_quat[gid]),
            "size": _float_array(model.geom_size[gid]),
            "friction": _float_array(model.geom_friction[gid]),
            "solref": _float_array(model.geom_solref[gid]),
            "solimp": _float_array(model.geom_solimp[gid]),
            "margin": float(model.geom_margin[gid]),
            "gap": float(model.geom_gap[gid]),
            "contype": int(model.geom_contype[gid]),
            "conaffinity": int(model.geom_conaffinity[gid]),
            "group": int(model.geom_group[gid]),
        })

    meshes = []
    for mid in range(model.nmesh):
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        meshes.append({
            "name": _name(model, mujoco.mjtObj.mjOBJ_MESH, mid),
            "vertex_count": vn,
            "face_count": fn,
            "vertex_sha256": _array_digest(model.mesh_vert[va:va + vn], "<f8"),
            "face_sha256": _array_digest(model.mesh_face[fa:fa + fn], "<i4"),
        })

    sites = []
    for sid in range(model.nsite):
        sites.append({
            "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, sid),
            "bodyid": int(model.site_bodyid[sid]),
            "pos": _float_array(model.site_pos[sid]),
            "quat": _float_array(model.site_quat[sid]),
            "size": _float_array(model.site_size[sid]),
            "type": int(model.site_type[sid]),
        })

    equalities = []
    for eid in range(model.neq):
        equalities.append({
            "name": _name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eid),
            "type": int(model.eq_type[eid]),
            "obj1id": int(model.eq_obj1id[eid]),
            "obj2id": int(model.eq_obj2id[eid]),
            "data": _float_array(model.eq_data[eid]),
            "solref": _float_array(model.eq_solref[eid]),
            "solimp": _float_array(model.eq_solimp[eid]),
            "active": int(model.eq_active0[eid]),
        })

    actuators = []
    for aid in range(model.nu):
        actuators.append({
            "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
            "trnid": [int(x) for x in model.actuator_trnid[aid]],
            "gear": _float_array(model.actuator_gear[aid]),
            "gainprm": _float_array(model.actuator_gainprm[aid]),
            "biasprm": _float_array(model.actuator_biasprm[aid]),
            "dynprm": _float_array(model.actuator_dynprm[aid]),
            "forcerange": _float_array(model.actuator_forcerange[aid]),
            "ctrlrange": _float_array(model.actuator_ctrlrange[aid]),
            "ctrllimited": int(model.actuator_ctrllimited[aid]),
            "forcelimited": int(model.actuator_forcelimited[aid]),
        })

    xml = ET.parse(xml_path).getroot()
    exclusions = sorted(
        sorted((node.get("body1", ""), node.get("body2", "")))
        for node in xml.findall("./contact/exclude")
    )
    option = model.opt
    return {
        "mujoco_version": mujoco.__version__,
        "dimensions": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "neq": int(model.neq), "nbody": int(model.nbody), "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh), "nsite": int(model.nsite),
        },
        "option": {
            "timestep": float(option.timestep),
            "integrator": int(option.integrator),
            "solver": int(option.solver),
            "iterations": int(option.iterations),
            "tolerance": float(option.tolerance),
            "cone": int(option.cone),
            "gravity": _float_array(option.gravity),
        },
        "joints": joints,
        "bodies": bodies,
        "geoms": geoms,
        "meshes": meshes,
        "sites": sites,
        "equalities": equalities,
        "actuators": actuators,
        "collision_exclusions": exclusions,
    }


def signature_sha256(signature: dict) -> str:
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
