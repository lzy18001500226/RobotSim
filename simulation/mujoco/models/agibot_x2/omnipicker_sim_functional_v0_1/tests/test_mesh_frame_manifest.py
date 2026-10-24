from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import unittest


ROOT = Path(__file__).resolve().parents[6]
PACKAGE = ROOT / "simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1"
SOURCE_ROOT = ROOT / "reference/AgibotTech/agibot_x2_urdf/X2_URDF-v1.4.0"
EXPECTED_FRAMES = {
    "narrow": ("R_hand_narrow3_Link", "narrow3_Link.stl"),
    "wide": ("R_hand_wide3_Link", "wide3_Link.stl"),
    "narrow_loop": ("R_hand_narrow_loop_Link", "narrow_loop_Link.stl"),
    "wide_loop": ("R_hand_wide_loop_Link", "wide_loop_Link.stl"),
}


def _floats(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split()) if value else default


def _quat_wxyz_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


class MeshFrameManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((PACKAGE / "visual_frames.json").read_text(encoding="utf-8"))
        cls.model_root = ET.parse(PACKAGE / "model.xml").getroot()
        cls.source_root = ET.parse(SOURCE_ROOT / "X2-Ultra_omnipicker.urdf").getroot()

    def test_manifest_covers_four_cad_supported_frames_and_pinned_meshes(self) -> None:
        self.assertEqual(
            self.manifest["schema_version"],
            "robotsim.x2-omnipicker-mesh-frames.v2",
        )
        self.assertEqual(self.manifest["classification"], "CAD_SUPPORTED_EXPERIMENTAL")
        self.assertEqual(set(self.manifest["frames"]), set(EXPECTED_FRAMES))
        self.assertEqual(
            self.manifest["source"]["cad_reference"]["sha256"],
            "5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b",
        )
        self.assertEqual(
            self.manifest["source"]["cad_reference"]["provenance_status"],
            "local_only_unverified",
        )
        self.assertEqual(
            self.manifest["source"]["cad_reference"]["step_component_to_urdf_link_identity"],
            "unresolved",
        )

        source_links = {link.get("name"): link for link in self.source_root.findall("link")}
        for frame_key, (link_name, mesh_name) in EXPECTED_FRAMES.items():
            with self.subTest(frame=frame_key):
                frame = self.manifest["frames"][frame_key]
                self.assertEqual(frame["link"], link_name)
                self.assertEqual(frame["mesh"], mesh_name)
                self.assertEqual(frame["classification"], "CAD_SUPPORTED_EXPERIMENTAL")
                mesh_path = SOURCE_ROOT / "meshes" / mesh_name
                self.assertEqual(
                    hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                    frame["mesh_sha256"],
                )

                link = source_links[link_name]
                for geometry in ("visual", "collision"):
                    origin = link.find(f"{geometry}/origin")
                    xyz = _floats(origin.get("xyz", "") if origin is not None else "", (0.0, 0.0, 0.0))
                    rpy = _floats(origin.get("rpy", "") if origin is not None else "", (0.0, 0.0, 0.0))
                    self.assertEqual(xyz, tuple(frame["source_urdf_origin_xyz_m"]))
                    self.assertEqual(rpy, tuple(frame["source_urdf_origin_rpy_rad"]))
                    self.assertEqual(rpy, (0.0, 0.0, 0.0))

    def test_manifest_matches_every_visual_and_collision_geom_transform(self) -> None:
        geoms = self.model_root.findall(".//geom")
        manifested_links = {frame["link"] for frame in self.manifest["frames"].values()}
        model_links = {
            geom.get("name", "").split("__", 1)[0]
            for geom in geoms
            if "__visual__" in geom.get("name", "")
            and geom.get("name", "").split("__", 1)[0] in {item[0] for item in EXPECTED_FRAMES.values()}
        }
        self.assertEqual(model_links, manifested_links)

        checked_geom_names: set[str] = set()
        for frame_key, frame in self.manifest["frames"].items():
            link_name = frame["link"]
            matching = [geom for geom in geoms if geom.get("name", "").startswith(link_name + "__")]
            visual = [geom for geom in matching if "__visual__" in geom.get("name", "")]
            collision = [geom for geom in matching if "__collision__" in geom.get("name", "")]
            self.assertEqual(len(visual), 1, frame_key)
            self.assertGreater(len(collision), 0, frame_key)
            expected_pos = tuple(frame["origin_xyz_m"])
            expected_quat = _quat_wxyz_from_rpy(tuple(frame["origin_rpy_rad"]))

            for geom in matching:
                checked_geom_names.add(geom.get("name", ""))
                with self.subTest(frame=frame_key, geom=geom.get("name")):
                    actual_pos = _floats(geom.get("pos", ""), (0.0, 0.0, 0.0))
                    actual_quat = _floats(geom.get("quat", ""), (1.0, 0.0, 0.0, 0.0))
                    self.assertEqual(actual_pos, expected_pos)
                    self.assertEqual(len(actual_quat), 4)
                    for actual, expected in zip(actual_quat, expected_quat):
                        self.assertAlmostEqual(actual, expected, delta=1e-12)

        self.assertEqual(len(checked_geom_names), 65)


if __name__ == "__main__":
    unittest.main()
