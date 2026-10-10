from __future__ import annotations

import copy
import unittest

from scripts.research.issue46_dual_omnihand_platform import validate_morphology_records


def fixture_records():
    links = {}
    for side in ("L", "R"):
        for family in ("thumb", "index", "middle", "ring", "pinky"):
            name = f"{side}_{family}_link"
            mesh = {
                "role": "visual",
                "shape": "mesh",
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "mesh_path": f"/official/{side}_{family}.stl",
                "mesh_scale": [1.0, 1.0, 1.0],
            }
            collision = {
                "role": "collision",
                "shape": "box",
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "size": [0.01, 0.02, 0.03],
            }
            links[name] = {"visuals": [mesh], "collisions": [collision]}
    joints = {
        "L_thumb_mcp_joint": {
            "type": "revolute",
            "parent": "L_thumb_link",
            "child": "L_index_link",
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "axis": [1.0, 0.0, 0.0],
            "range": [0.0, 1.0],
        },
        "L_thumb_dip_joint": {
            "type": "revolute",
            "parent": "L_index_link",
            "child": "L_middle_link",
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "axis": [1.0, 0.0, 0.0],
            "range": [0.0, 1.0],
        },
    }
    mimic = {"L_thumb_dip_joint": {"driver": "L_thumb_mcp_joint", "multiplier": 1.3, "offset": 0.0}}
    return links, copy.deepcopy(links), joints, copy.deepcopy(joints), mimic, copy.deepcopy(mimic)


class DualOmniHandMorphologyTests(unittest.TestCase):
    def test_official_visual_and_collision_records_match(self) -> None:
        records = fixture_records()
        self.assertEqual(validate_morphology_records(*records), [])

    def test_missing_visual_mesh_is_rejected(self) -> None:
        source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic = fixture_records()
        compiled["L_thumb_link"]["visuals"].clear()
        errors = validate_morphology_records(
            source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic
        )
        self.assertTrue(any("L_thumb_link visuals parity mismatch" in error for error in errors))
        self.assertTrue(any("L thumb visual mesh family missing" in error for error in errors))

    def test_missing_finger_joint_is_rejected(self) -> None:
        source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic = fixture_records()
        del compiled_joints["L_thumb_dip_joint"]
        errors = validate_morphology_records(
            source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic
        )
        self.assertTrue(any("missing compiled joint/topology record: L_thumb_dip_joint" == error for error in errors))

    def test_left_right_geometry_swap_is_rejected(self) -> None:
        source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic = fixture_records()
        compiled["L_thumb_link"]["visuals"], compiled["R_thumb_link"]["visuals"] = (
            compiled["R_thumb_link"]["visuals"],
            compiled["L_thumb_link"]["visuals"],
        )
        errors = validate_morphology_records(
            source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic
        )
        self.assertTrue(any("L_thumb_link visuals parity mismatch" in error for error in errors))
        self.assertTrue(any("R_thumb_link visuals parity mismatch" in error for error in errors))

    def test_dropped_mimic_relation_is_rejected(self) -> None:
        source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic = fixture_records()
        compiled_mimic.clear()
        errors = validate_morphology_records(
            source, compiled, source_joints, compiled_joints, source_mimic, compiled_mimic
        )
        self.assertTrue(any("mimic relation set mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
