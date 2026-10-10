"""Fixture-only proof of concept for model fidelity review checks."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET


def _assert_one_expected_tool(wrist: ET.Element, expected: str, known_tool_roots: set[str]) -> None:
    tool_roots = [
        body.get("name")
        for body in wrist.findall("body")
        if body.get("name") in known_tool_roots
    ]
    if tool_roots != [expected]:
        raise AssertionError(f"expected only {expected}; found {tool_roots}")


def _assert_visual_and_collision_geometry(tool: ET.Element) -> None:
    visual_meshes = []
    collision_geoms = []
    for geom in tool.findall(".//geom"):
        collision_enabled = (
            int(geom.get("contype", "1")) != 0
            or int(geom.get("conaffinity", "1")) != 0
        )
        if collision_enabled:
            collision_geoms.append(geom)
        elif geom.get("mesh"):
            visual_meshes.append(geom)
    if not visual_meshes:
        raise AssertionError("tool has no mesh-backed visual geom")
    if not collision_geoms:
        raise AssertionError("tool has no contact-enabled collision geom")


class ModelFidelityPocTests(unittest.TestCase):
    def test_replacement_tool_cannot_coexist_with_known_stock_hand_root(self) -> None:
        duplicate = ET.fromstring(
            """<body name="right_wrist">
              <body name="right_stock_hand"/><body name="right_parallel_gripper"/>
            </body>"""
        )
        with self.assertRaisesRegex(AssertionError, "found "):
            _assert_one_expected_tool(
                duplicate, "right_parallel_gripper",
                {"right_stock_hand", "right_parallel_gripper"},
            )

        replacement_only = ET.fromstring(
            """<body name="right_wrist"><body name="right_parallel_gripper"/></body>"""
        )
        _assert_one_expected_tool(
            replacement_only, "right_parallel_gripper",
            {"right_stock_hand", "right_parallel_gripper"},
        )

    def test_visual_mesh_and_primitive_collision_are_checked_separately(self) -> None:
        faithful_visual_proxy = ET.fromstring(
            """<body name="tool">
              <geom name="visual" type="mesh" mesh="hand_mesh" contype="0" conaffinity="0"/>
              <geom name="collision" type="box" size="0.1 0.1 0.1"/>
            </body>"""
        )
        _assert_visual_and_collision_geometry(faithful_visual_proxy)

        collision_only = ET.fromstring(
            """<body name="tool">
              <geom name="collision_a" type="box" size="0.1 0.1 0.1"/>
              <geom name="collision_b" type="capsule" size="0.02 0.1"/>
            </body>"""
        )
        with self.assertRaisesRegex(AssertionError, "mesh-backed visual"):
            _assert_visual_and_collision_geometry(collision_only)


if __name__ == "__main__":
    unittest.main()
