from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from simulation.mujoco.models.agibot_x2 import X2OmniPickerModelError, load_x2_omnipicker
from simulation.mujoco.models.agibot_x2.loader import (
    CAD_MODEL_ID,
    STL_MODEL_ID,
    _resolve_model_file,
    _verify_collision_assets,
    _verify_file_sha256,
    selected_x2_omnipicker_model_id,
)


ROOT = Path(__file__).resolve().parents[6]
PACKAGE = ROOT / "simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1"


class RegistryContractTests(unittest.TestCase):
    def test_registry_exposes_only_cad_and_stl_with_one_central_default(self) -> None:
        registry = json.loads((PACKAGE.parent / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(list(registry["models"]), [CAD_MODEL_ID, STL_MODEL_ID])
        self.assertEqual(registry["default_model_id"], STL_MODEL_ID)
        self.assertEqual(registry["selection_status"], "PENDING_HUMAN_FULL_X2_REVIEW")
        self.assertEqual(selected_x2_omnipicker_model_id(repo_root=ROOT), STL_MODEL_ID)

    def test_obsolete_candidate_alias_fails_without_fallback(self) -> None:
        with self.assertRaisesRegex(X2OmniPickerModelError, "unknown X2 OmniPicker model ID"):
            load_x2_omnipicker("Candidate B", repo_root=ROOT)
        with self.assertRaisesRegex(X2OmniPickerModelError, "unknown X2 OmniPicker model ID"):
            load_x2_omnipicker("X2-OmniPicker-Sim-Functional-v0.1", repo_root=ROOT)

    def test_missing_registered_model_fails_without_searching_legacy_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing_package = Path(temp) / "missing-package"
            with self.assertRaisesRegex(X2OmniPickerModelError, "missing registered model file"):
                _resolve_model_file(missing_package, "model.xml", "0" * 64)

    def test_wrong_vendor_mesh_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / "source_mesh.stl"
            mesh.write_bytes(b"not the registered vendor mesh")
            with self.assertRaisesRegex(X2OmniPickerModelError, "SHA-256 mismatch"):
                _verify_file_sha256(mesh, "0" * 64, "vendor mesh")

    def test_unreferenced_collision_proxy_hash_is_still_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            collision_dir = package / "collision_generated"
            collision_dir.mkdir()
            referenced = collision_dir / "referenced.stl"
            unreferenced = collision_dir / "unreferenced.stl"
            referenced.write_bytes(b"referenced collision geometry")
            unreferenced.write_bytes(b"unreferenced registered collision geometry")
            expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (referenced, unreferenced)}
            _verify_collision_assets(package, collision_dir, expected)
            unreferenced.write_bytes(b"altered but not referenced by this MJCF")
            with self.assertRaisesRegex(X2OmniPickerModelError, "SHA-256 mismatch"):
                _verify_collision_assets(package, collision_dir, expected)

    def test_collision_proxy_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            collision_dir = package / "collision_generated"
            collision_dir.mkdir()
            external = package / "outside.stl"
            external.write_bytes(b"outside")
            (collision_dir / "proxy.stl").symlink_to(external)
            expected = {"proxy.stl": hashlib.sha256(external.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(X2OmniPickerModelError, "non-regular or linked"):
                _verify_collision_assets(package, collision_dir, expected)

    def test_missing_collision_proxy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            collision_dir = package / "collision_generated"
            collision_dir.mkdir()
            with self.assertRaisesRegex(X2OmniPickerModelError, "incomplete"):
                _verify_collision_assets(package, collision_dir, {"missing.stl": "0" * 64})

    def test_registry_metadata_preserves_cad_and_stl_claim_boundaries(self) -> None:
        registry = json.loads((PACKAGE.parent / "registry.json").read_text(encoding="utf-8"))
        cad = registry["models"][CAD_MODEL_ID]
        stl = registry["models"][STL_MODEL_ID]
        self.assertEqual(cad["frozen_pr90_reference_model_sha256"], "32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45")
        self.assertEqual(cad["qualification"], "isolated_unloaded_single_actuator_open_close")
        self.assertIn("bottle_grasp", cad["unsupported_claims"])
        self.assertEqual(cad["fallback_policy"], "NONE_FAIL_CLOSED")
        self.assertEqual(stl["classification"], "KINEMATIC_SIMULATION_ONLY")
        self.assertEqual(stl["model_dimensions"], {"nq": 6, "nv": 6, "nu": 0, "neq": 0})
        self.assertIn("dynamic_passive_link_physics", stl["unsupported_claims"])
        self.assertEqual(stl["fallback_policy"], "NONE_FAIL_CLOSED")

    def test_static_signature_fixture_matches_cad_portability_lock(self) -> None:
        portability = json.loads((PACKAGE / "portability.json").read_text(encoding="utf-8"))
        fixture = PACKAGE / "tests/fixtures/pr90_model_signature.json"
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        self.assertEqual(digest, portability["static_signature_fixture_sha256"])

    def test_explicit_repo_root_uses_its_own_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".git").write_text("gitdir: fixture\n", encoding="utf-8")
            registry_dir = root / "simulation/mujoco/models/agibot_x2"
            registry_dir.mkdir(parents=True)
            (registry_dir / "registry.json").write_text(json.dumps({"schema_version": "unsupported"}), encoding="utf-8")
            with self.assertRaisesRegex(X2OmniPickerModelError, "unsupported X2 OmniPicker registry schema"):
                load_x2_omnipicker(CAD_MODEL_ID, repo_root=root)


if __name__ == "__main__":
    unittest.main()
