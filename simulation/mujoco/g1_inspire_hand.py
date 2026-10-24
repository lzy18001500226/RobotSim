"""Official DFQ source-joint mapping used by the G1 M0 overlay."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

URDF = Path()
SIDES = ("L", "R")
CHANNELS = (
    ("pinky_proximal", "pinky_proximal"),
    ("ring_proximal", "ring_proximal"),
    ("middle_proximal", "middle_proximal"),
    ("index_proximal", "index_proximal"),
    ("thumb_proximal_pitch", "thumb_pitch"),
    ("thumb_proximal_yaw", "thumb_yaw"),
)


def parse_urdf() -> tuple[dict, dict, dict, dict]:
    root = ET.parse(URDF).getroot()
    joints = {node.get("name"): node for node in root.findall("joint")}
    limits, mimics, velocities = {}, {}, {}
    for name, joint in joints.items():
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None:
            limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
            if limit.get("velocity") is not None:
                velocities[name] = float(limit.get("velocity"))
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics[name] = {
                "follower": name,
                "parent": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", "1")),
                "offset": float(mimic.get("offset", "0")),
            }
    return joints, limits, mimics, velocities


def channel_joint(side: str, suffix: str) -> str:
    if suffix == "thumb_pitch":
        return f"{side}_thumb_proximal_pitch_joint"
    if suffix == "thumb_yaw":
        return f"{side}_thumb_proximal_yaw_joint"
    return f"{side}_{suffix}_joint"


def derive_channel_ranges(limits: dict, mimics: dict) -> dict:
    result = {}
    for side in SIDES:
        for channel, suffix in CHANNELS:
            name = channel_joint(side, suffix)
            lo, hi = limits[name]
            raw = (lo, hi)
            for relation in mimics.values():
                if relation["parent"] != name:
                    continue
                f_lo, f_hi = limits[relation["follower"]]
                mult, offset = relation["multiplier"], relation["offset"]
                if mult == 0:
                    if not f_lo <= offset <= f_hi:
                        raise ValueError(f"Mimic offset violates follower limit: {relation}")
                else:
                    a, b = (f_lo - offset) / mult, (f_hi - offset) / mult
                    lo = max(lo, min(a, b))
                    hi = min(hi, max(a, b))
            if lo > hi:
                raise ValueError(f"No feasible command range for {name}")
            result[name] = {
                "channel": channel,
                "official_urdf_range_rad": list(raw),
                "simulation_feasible_range_rad": [lo, hi],
                "close_target_rad": hi,
                "open_target_rad": lo,
                "derivation": "intersection of official driver and mimic-follower limits",
            }
    return result


def normalized_target(side: str, channel: str, u: float, ranges: dict) -> float:
    suffix = dict(CHANNELS)[channel]
    lo, hi = ranges[channel_joint(side, suffix)]["simulation_feasible_range_rad"]
    return hi - float(u) * (hi - lo)
