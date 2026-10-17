"""Reusable table and bottle definitions for manipulation scene prototypes."""

from __future__ import annotations

import mujoco


G1_CANONICAL_TABLE_CENTER_XY = (0.300, -0.100)
G1_CANONICAL_TABLE_HALF_EXTENTS = (0.200, 0.200)
G1_CANONICAL_TABLE_TOP_Z = 0.800
G1_CANONICAL_TABLE_RGBA = (0.6, 0.4, 0.2, 1.0)

CANONICAL_X2_BOTTLE_START_BODY_POS = (0.300, 0.000, 0.9175)
CANONICAL_X2_BOTTLE_TARGET_BODY_POS = (0.300, -0.200, 0.9175)
CANONICAL_X2_BOTTLE_DIAMETER_M = 0.070
CANONICAL_X2_BOTTLE_HEIGHT_M = 0.2445
CANONICAL_X2_BOTTLE_MASS_KG = 0.570
CANONICAL_X2_BOTTLE_GEOMETRIC_Z_BOUNDS_M = (-0.1175, 0.127)
CANONICAL_X2_BOTTLE_GEOMS = (
    {
        "name": "bottle_body",
        "type": "cylinder",
        "pos": (0.0, 0.0, -0.04),
        "size": (0.035, 0.0775, 0.0),
        "mass": 0.49,
        "rgba": (0.12, 0.52, 0.82, 1.0),
    },
    {
        "name": "bottle_shoulder",
        "type": "ellipsoid",
        "pos": (0.0, 0.0, 0.0535),
        "size": (0.035, 0.035, 0.026),
        "mass": 0.05,
        "rgba": (0.12, 0.52, 0.82, 1.0),
    },
    {
        "name": "bottle_neck",
        "type": "cylinder",
        "pos": (0.0, 0.0, 0.0855),
        "size": (0.018, 0.020, 0.0),
        "mass": 0.02,
        "rgba": (0.12, 0.52, 0.82, 1.0),
    },
    {
        "name": "bottle_cap",
        "type": "cylinder",
        "pos": (0.0, 0.0, 0.1185),
        "size": (0.020, 0.0085, 0.0),
        "mass": 0.01,
        "rgba": (0.10, 0.16, 0.21, 1.0),
    },
)


def add_g1_canonical_table(spec: mujoco.MjSpec) -> None:
    """Add the table body and dimensions from the accepted G1 M0 scene."""
    table = spec.worldbody.add_body(
        name="m0_table",
        pos=[*G1_CANONICAL_TABLE_CENTER_XY, 0.0],
    )
    table.add_geom(
        name="m0_table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.775],
        size=[0.200, 0.200, 0.025],
        rgba=G1_CANONICAL_TABLE_RGBA,
        friction=[1.2, 0.01, 0.001],
        condim=4,
        group=1,
    )
    for name, x, y in (
        ("m0_table_leg_front_left", -0.175, 0.175),
        ("m0_table_leg_front_right", 0.175, 0.175),
        ("m0_table_leg_back_left", -0.175, -0.175),
        ("m0_table_leg_back_right", 0.175, -0.175),
    ):
        table.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, 0.375],
            size=[0.025, 0.025, 0.375],
            rgba=G1_CANONICAL_TABLE_RGBA,
            friction=[0.9, 0.01, 0.001],
            group=1,
        )
    table.add_site(
        name="m0_target_site",
        pos=[0.0, 0.0, 0.805],
        size=[0.005, 0.005, 0.005],
        rgba=[0.0, 0.0, 0.0, 0.0],
    )
