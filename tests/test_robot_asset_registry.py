import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_robot_asset_registry.py"
SPEC = importlib.util.spec_from_file_location("asset_registry_checker", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(checker)


class RobotAssetRegistryTests(unittest.TestCase):
    def test_manifest_hash_changes_when_identity_record_changes(self):
        base = {"assets": [{"asset_id": "x2_ultra_omnipicker_v1_4"}]}
        hashes = {"profiles": {}}
        compiled = {"profiles": {}}
        original = checker.canonical_manifest_hash(base, hashes, compiled)
        changed = checker.canonical_manifest_hash(
            {"assets": [{"asset_id": "x2_ultra_omnihand_v1_4"}]}, hashes, compiled
        )
        self.assertNotEqual(original, changed)

    def test_omnipicker_fails_omnihand_expectations(self):
        expected = {
            "expected_structure": {"njnt": 63, "ngeom": 144, "nmesh": 51},
            "required_joint_tokens": ["thumb", "index", "middle", "ring", "little"],
            "forbidden_joint_tokens": ["claw_joint"],
        }
        actual = {
            "signature": {"njnt": 35, "ngeom": 106, "nmesh": 50},
            "joint_names": ["left_claw_joint", "right_claw_joint"],
            "body_names": ["L_omnipicker_base_link", "R_omnipicker_base_link"],
            "geom_names": [], "actuator_names": [],
        }
        errors = checker.compare_model(expected, actual)
        self.assertTrue(any("compiled njnt mismatch" in error for error in errors))
        self.assertTrue(any("required model name token missing: thumb" in error for error in errors))
        self.assertTrue(any("claw_joint" in error for error in errors))

    def test_omnihand_fails_omnipicker_expectations(self):
        expected = {
            "expected_structure": {"njnt": 35, "ngeom": 106, "nmesh": 50},
            "required_joint_tokens": ["left_claw_joint", "right_claw_joint"],
            "forbidden_joint_tokens": ["R_thumb_", "L_index_"],
        }
        actual = {
            "signature": {"njnt": 63, "ngeom": 144, "nmesh": 51},
            "joint_names": ["R_thumb_proximal_joint", "L_index_proximal_joint"],
            "body_names": [], "geom_names": [], "actuator_names": [],
        }
        errors = checker.compare_model(expected, actual)
        self.assertTrue(any("compiled njnt mismatch" in error for error in errors))
        self.assertTrue(any("required model name token missing: left_claw_joint" in error for error in errors))
        self.assertTrue(any("forbidden/mutually-exclusive" in error for error in errors))

    def test_name_token_mismatch_is_reported(self):
        expected = {"expected_structure": {}, "required_joint_tokens": ["R_thumb"], "forbidden_joint_tokens": []}
        actual = {
            "signature": {}, "joint_names": ["r_thumb"], "body_names": [],
            "geom_names": [], "actuator_names": [],
        }
        self.assertTrue(checker.compare_model(expected, actual))

    def test_exact_names_report_missing_and_unexpected_geometry(self):
        expected = {
            "expected_structure": {}, "required_joint_tokens": [], "forbidden_joint_tokens": []
        }
        actual = {
            "mujoco_version": "3.3.6", "signature": {}, "joint_names": ["driver"],
            "body_names": [], "geom_names": ["wrong_collision"], "actuator_names": [], "mesh_names": [],
        }
        reference = {
            "mujoco_version": "3.3.6", "snapshot": {
                "joint_names": ["driver", "follower"], "body_names": [],
                "geom_names": ["expected_collision"], "actuator_names": [], "mesh_names": [],
            }
        }
        errors = checker.compare_model(expected, actual, reference)
        self.assertIn("compiled joint_names missing exact names: follower", errors)
        self.assertIn("compiled geom_names missing exact names: expected_collision", errors)
        self.assertIn("compiled geom_names has unexpected exact names: wrong_collision", errors)


if __name__ == "__main__":
    unittest.main()
