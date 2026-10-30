#!/usr/bin/env python3
"""Focused geometry tests for the static RGB-D capture overlay."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


MODULE_PATH = Path(__file__).with_name("issue46_rgbd_truth_capture.py")
SPEC = importlib.util.spec_from_file_location("rgbd_truth_capture", MODULE_PATH)
assert SPEC and SPEC.loader
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


class RGBDTruthCaptureGeometryTests(unittest.TestCase):
    def test_look_at_points_local_minus_z_to_target(self) -> None:
        position = np.array([0.4, -0.8, 1.2])
        target = np.array([1.2, -0.1, 0.6])
        rotation = CAPTURE.look_at_rotation(position, target)
        expected_forward = (target - position) / np.linalg.norm(target - position)
        np.testing.assert_allclose(-rotation[:, 2], expected_forward, atol=1e-10)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)

    def test_parent_local_pose_recovers_requested_world_pose(self) -> None:
        parent_rotation = CAPTURE.look_at_rotation(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        parent_position = np.array([0.7, -0.2, 0.4])
        camera_position = np.array([1.0, 0.3, 1.1])
        camera_rotation = CAPTURE.look_at_rotation(camera_position, np.array([1.7, 0.1, 0.9]))
        local_rotation, local_position = CAPTURE.pose_in_parent(
            parent_rotation,
            parent_position,
            camera_rotation,
            camera_position,
        )
        np.testing.assert_allclose(parent_rotation @ local_rotation, camera_rotation, atol=1e-10)
        np.testing.assert_allclose(parent_rotation @ local_position + parent_position, camera_position, atol=1e-10)

    def test_quaternion_is_unit_for_half_turn(self) -> None:
        rotation = np.diag([-1.0, -1.0, 1.0])
        quaternion = CAPTURE.quaternion_wxyz(rotation)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)

    def test_mask_bbox_handles_visible_and_empty_objects(self) -> None:
        mask = np.zeros((8, 9), dtype=bool)
        mask[2:6, 3:8] = True
        self.assertEqual(CAPTURE.mask_bbox_xyxy(mask), [3, 2, 7, 5])
        self.assertIsNone(CAPTURE.mask_bbox_xyxy(np.zeros((2, 2), dtype=bool)))

    def test_framebuffer_overlay_only_increases_requested_dimension(self) -> None:
        root = ET.fromstring("<mujoco><visual><global offwidth='640' offheight='480'/></visual></mujoco>")
        result = CAPTURE.ensure_offscreen_framebuffer(root, width=1280, height=720)
        self.assertEqual(result, {"offwidth": 1280, "offheight": 720})
        global_visual = root.find("./visual/global")
        self.assertEqual(global_visual.get("offwidth"), "1280")
        self.assertEqual(global_visual.get("offheight"), "720")

    def test_only_existing_canonical_reference_asset_is_rebound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root_path = Path(temporary)
            reference = root_path / "reference"
            reference.mkdir()
            temp_asset = reference / "issue46_rgbd_capture_test_asset.bin"
            temp_asset.write_bytes(b"test")
            proxy = root_path / "simulation" / "mujoco" / "proxy.stl"
            proxy.parent.mkdir(parents=True)
            proxy.write_bytes(b"proxy")
            root = ET.fromstring(
                "<mujoco><asset>"
                "<mesh file='/retired/worktree/reference/issue46_rgbd_capture_test_asset.bin'/>"
                "<mesh file='/retired/worktree/simulation/mujoco/proxy.stl'/>"
                "<mesh file='relative_missing.stl'/>"
                "</asset></mujoco>"
            )
            rebound = CAPTURE.rebind_missing_workspace_assets(
                root,
                canonical_reference=reference,
                canonical_root=root_path,
            )
            self.assertEqual(len(rebound), 2)
            meshes = root.findall(".//mesh")
            self.assertEqual(meshes[0].get("file"), str(temp_asset))
            self.assertEqual(meshes[1].get("file"), str(proxy))
            self.assertEqual(meshes[2].get("file"), "relative_missing.stl")


if __name__ == "__main__":
    unittest.main()
