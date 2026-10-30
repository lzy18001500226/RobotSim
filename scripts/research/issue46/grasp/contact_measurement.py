"""Reusable contact accounting for historical STL cylinder-lift experiments."""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}


def _body_name(model: Any, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body#{body_id}"


def _geom_name(model: Any, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"


def contact_data(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    """Measure cylinder support/contact forces without writing simulator state."""
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    if cylinder < 0:
        raise RuntimeError("missing test cylinder")
    records: list[dict[str, object]] = []
    pad_vectors = {name: np.zeros(3) for name in PADS}
    table_contacts: list[dict[str, object]] = []
    table_vector = np.zeros(3)
    nonpad: list[dict[str, object]] = []
    robot_table: list[dict[str, object]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        names = (_body_name(model, body1), _body_name(model, body2))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        world_force = np.asarray(contact.frame, dtype=float).reshape(3, 3).T @ wrench[:3]
        record: dict[str, object] = {
            "index": index,
            "body_names": names,
            "geom_names": (_geom_name(model, geom1), _geom_name(model, geom2)),
            "distance_m": float(contact.dist),
            "force_contact_frame_N_Nm": wrench.tolist(),
            "force_world_frame_N": world_force.tolist(),
        }
        records.append(record)
        if cylinder in (body1, body2):
            other = names[1] if body1 == cylinder else names[0]
            force_on_cylinder = world_force if body1 == cylinder else -world_force
            record["force_on_cylinder_world_N"] = force_on_cylinder.tolist()
            if other in PADS:
                pad_vectors[other] += force_on_cylinder
            elif other == "raicom_table":
                table_contacts.append(record)
                table_vector += force_on_cylinder
            elif other.startswith("R_hand_"):
                nonpad.append(record)
        if "raicom_table" in names and any(name.startswith(("R_hand_", "right_")) for name in names):
            robot_table.append(record)
    return {
        "records": records,
        "pad_forces_on_cylinder_world_N": {name: vector.tolist() for name, vector in pad_vectors.items()},
        "pad_force_norm_N": {name: float(np.linalg.norm(vector)) for name, vector in pad_vectors.items()},
        "table_force_on_cylinder_world_N": table_vector.tolist(),
        "table_contacts": table_contacts,
        "nonpad_cylinder_contacts": nonpad,
        "robot_table_contacts": robot_table,
    }
