"""Original-STL gripping-surface rays used by the 101-pose static regression."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import mujoco


APERTURE_X_M = 0.0
SECTION_Z_M = (0.110, 0.1125, 0.115, 0.1175, 0.120)


def _stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"not a binary STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError(f"malformed binary STL: {path}")
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    return np.frombuffer(data[84:], dtype=dtype)["vertices"].astype(np.float64)


def _origin(xyz_value, rpy_value) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_value, dtype=np.float64)
    roll, pitch, yaw = np.asarray(rpy_value, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return xyz, rotation


def _ray_y(triangles: np.ndarray, x: float, z: float, side: str) -> float | None:
    origin = np.array([x, -0.5, z], dtype=np.float64)
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.array([0.0, 1.0, 0.0]), e2)
    det = np.einsum("ij,ij->i", e1, h)
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=np.abs(det) > 1e-14)
    s = origin - triangles[:, 0]
    u = inv * np.einsum("ij,ij->i", s, h)
    cross = np.cross(s, e1)
    v = inv * cross[:, 1]
    t = inv * np.einsum("ij,ij->i", e2, cross)
    ids = np.flatnonzero(
        (np.abs(det) > 1e-14)
        & (u >= -1e-9)
        & (v >= -1e-9)
        & (u + v <= 1 + 1e-9)
        & (t > 0)
    )
    if not len(ids):
        return None
    ys = -0.5 + t[ids]
    return float(ys[np.argmax(ys) if side == "narrow" else np.argmin(ys)])


class OriginalSTLSurfaces:
    def __init__(self, model: mujoco.MjModel, source_root: Path):
        self.model = model
        package = Path(__file__).resolve().parents[1]
        frame_manifest = json.loads((package / "visual_frames.json").read_text(encoding="utf-8"))
        self.meshes = {}
        for side, frame in frame_manifest["frames"].items():
            link_name = frame["link"]
            mesh_path = source_root / "X2_URDF-v1.4.0/meshes" / frame["mesh"]
            if not mesh_path.is_file():
                raise ValueError(f"missing original visual STL: {mesh_path}")
            geom_name = f"{link_name}__visual__{side}3_Link"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                raise ValueError(f"missing visual geom {geom_name}")
            body_id = int(model.geom_bodyid[geom_id])
            pos, rotation = _origin(frame["origin_xyz_m"], frame["origin_rpy_rad"])
            self.meshes[side] = {
                "triangles": _stl_triangles(mesh_path),
                "pos": pos,
                "rotation": rotation,
                "body_id": body_id,
            }

    def _world_triangles(self, data: mujoco.MjData, side: str) -> np.ndarray:
        record = self.meshes[side]
        triangles = record["triangles"]
        pos = record["pos"]
        rotation = record["rotation"]
        body = record["body_id"]
        link_frame = triangles @ rotation.T + pos
        return link_frame @ data.xmat[body].reshape(3, 3).T + data.xpos[body]

    def five_section_widths_mm(self, data: mujoco.MjData) -> list[float]:
        triangles = {side: self._world_triangles(data, side) for side in ("narrow", "wide")}
        widths = []
        for z in SECTION_Z_M:
            narrow = _ray_y(triangles["narrow"], APERTURE_X_M, z, "narrow")
            wide = _ray_y(triangles["wide"], APERTURE_X_M, z, "wide")
            if narrow is None or wide is None:
                raise ValueError(f"source STL aperture ray did not intersect both jaws at z={z}")
            widths.append((wide - narrow) * 1000.0)
        return widths
