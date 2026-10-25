from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from simulation.mujoco.models.agibot_x2.omnipicker_sim_functional_v0_1 import provision_collision


class CollisionProvisioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name) / "package"
        self.package.mkdir()
        self.source = self.package / "source"
        self.source.mkdir()
        self.output = self.package / "collision_generated"
        self.manifest_path = self.package / "collision_proxy_manifest.json"
        self.source_manifest_path = self.package / "source_assets.json"
        self.source_manifest_path.write_text("{}\n", encoding="utf-8")
        self.unrelated = self.package / "unrelated.txt"
        self.unrelated.write_bytes(b"preserve this sibling file")
        self.meshes = self._make_meshes()

        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(provision_collision, "OUTPUT", self.output))
        self.patches.enter_context(patch.object(provision_collision, "PROXY_MANIFEST", self.manifest_path))
        self.patches.enter_context(patch.object(provision_collision, "SOURCE_MANIFEST", self.source_manifest_path))
        self.patches.enter_context(
            patch.object(
                provision_collision,
                "source_root",
                return_value=(self.source, {"commit": "fixture-pinned-commit"}),
            )
        )
        self.patches.enter_context(
            patch.object(
                provision_collision.importlib.metadata,
                "version",
                side_effect=lambda name: {
                    "coacd": provision_collision.EXPECTED_COACD,
                    "numpy": provision_collision.EXPECTED_NUMPY,
                }[name],
            )
        )
        self.patches.enter_context(
            patch.object(provision_collision.np, "__version__", provision_collision.EXPECTED_NUMPY)
        )

        proxy_manifest = {
            "version": provision_collision.EXPECTED_COACD,
            "settings": {"seed": 17},
            "mesh_decompositions": [mesh["manifest"] for mesh in self.meshes],
        }
        self.manifest_path.write_text(json.dumps(proxy_manifest), encoding="utf-8")
        self.expected_bytes = {
            name: content
            for mesh in self.meshes
            for name, content in mesh["output_bytes"].items()
        }

    def _make_meshes(self) -> list[dict]:
        result = []
        dtype = np.dtype(
            [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
        )
        for index, (stem, part_count) in enumerate((("mesh_a", 80), ("mesh_b", 81))):
            base = float(index * 2)
            source_vertices = np.asarray(
                [[base, 0, 0], [base + 1, 0, 0], [base, 1, 0]], dtype=np.float32
            )
            triangles = np.zeros(1, dtype=dtype)
            triangles["vertices"][0] = source_vertices
            source_path = self.source / f"{stem}.stl"
            source_path.write_bytes(b"fixture source mesh".ljust(80, b" ") + struct.pack("<I", 1) + triangles.tobytes())
            vertices, inverse = np.unique(source_vertices, axis=0, return_inverse=True)
            faces = inverse.reshape(1, 3).astype(np.uint32)
            parts = []
            output_bytes = {}
            for part_index in range(part_count):
                filename = f"{stem}_coacd_{part_index:02d}.stl"
                record_name = f"fixture source-derived piece {stem}-{part_index:03d}"
                piece_bytes = provision_collision.write_binary_stl(vertices, faces, record_name)
                parts.append(
                    {
                        "filename": filename,
                        "name": record_name,
                        "sha256": hashlib.sha256(piece_bytes).hexdigest(),
                    }
                )
                output_bytes[filename] = piece_bytes
            result.append(
                {
                    "vertices": vertices,
                    "faces": faces,
                    "output_bytes": output_bytes,
                    "manifest": {
                        "source_mesh": source_path.name,
                        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "vertices_unique": 3,
                        "triangles": 1,
                        "parts": parts,
                    },
                }
            )
        return result

    def fake_coacd(self, *, fail_on_call: int | None = None, on_call=None):
        calls = 0

        def run_coacd(_mesh, **_settings):
            nonlocal calls
            calls += 1
            if on_call is not None:
                on_call(calls)
            if calls == fail_on_call:
                raise RuntimeError("injected CoACD failure")
            mesh = self.meshes[calls - 1]
            return [
                (mesh["vertices"], mesh["faces"])
                for _ in mesh["manifest"]["parts"]
            ]

        module = types.ModuleType("coacd")
        module.Mesh = lambda vertices, faces: (vertices, faces)
        module.run_coacd = run_coacd
        return module

    def provision_with(self, fake_module) -> dict:
        with patch.dict(sys.modules, {"coacd": fake_module}):
            return provision_collision.provision()

    def stage_directories(self) -> list[Path]:
        return list(self.package.glob(".collision_generated.staging-*"))

    def assert_output_matches_fixture(self) -> None:
        self.assertEqual({path.name for path in self.output.iterdir()}, set(self.expected_bytes))
        for name, expected in self.expected_bytes.items():
            self.assertEqual((self.output / name).read_bytes(), expected)

    def test_success_generates_then_atomically_publishes_complete_hash_verified_output(self):
        def observe_generation(call: int) -> None:
            self.assertFalse(self.output.exists())
            stages = self.stage_directories()
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0].parent, self.output.parent)

        original_publish = provision_collision._rename_directory_noreplace

        def observe_publication(staging: Path, destination: Path) -> None:
            self.assertEqual(destination, self.output)
            self.assertFalse(self.output.exists())
            self.assertEqual({path.name for path in staging.iterdir()}, set(self.expected_bytes))
            original_publish(staging, destination)

        with patch.object(provision_collision, "_rename_directory_noreplace", side_effect=observe_publication):
            result = self.provision_with(self.fake_coacd(on_call=observe_generation))

        self.assertEqual(result["status"], "GENERATED_AND_HASH_VERIFIED")
        self.assertEqual(result["piece_count"], 161)
        self.assert_output_matches_fixture()
        self.assertEqual(self.stage_directories(), [])

    def test_partial_coacd_failure_cleans_own_stage_and_retry_succeeds(self):
        orphan = self.package / ".collision_generated.staging-orphan"
        orphan.mkdir()
        orphan_marker = orphan / "partial.stl"
        orphan_marker.write_bytes(b"preserve orphan from an earlier interrupted process")

        def fail_after_first_piece(call: int) -> None:
            if call == 2:
                stages = [path for path in self.stage_directories() if path != orphan]
                self.assertEqual(len(stages), 1)
                self.assertEqual(
                    len(list(stages[0].iterdir())),
                    len(self.meshes[0]["manifest"]["parts"]),
                )

        with self.assertRaisesRegex(RuntimeError, "injected CoACD failure"):
            self.provision_with(self.fake_coacd(fail_on_call=2, on_call=fail_after_first_piece))

        self.assertFalse(self.output.exists())
        self.assertEqual(self.stage_directories(), [orphan])
        self.assertEqual(self.unrelated.read_bytes(), b"preserve this sibling file")
        self.assertEqual(orphan_marker.read_bytes(), b"preserve orphan from an earlier interrupted process")

        result = self.provision_with(self.fake_coacd())
        self.assertEqual(result["status"], "GENERATED_AND_HASH_VERIFIED")
        self.assert_output_matches_fixture()
        self.assertEqual(self.stage_directories(), [orphan])
        self.assertEqual(orphan_marker.read_bytes(), b"preserve orphan from an earlier interrupted process")

    def test_complete_existing_output_is_reused_without_rerunning_coacd(self):
        self.provision_with(self.fake_coacd())

        def must_not_run(_mesh, **_settings):
            raise AssertionError("CoACD should not run when a complete exact output exists")

        module = self.fake_coacd()
        module.run_coacd = must_not_run
        result = self.provision_with(module)

        self.assertEqual(result["status"], "REUSED_HASH_VERIFIED")
        self.assert_output_matches_fixture()

    def test_existing_incomplete_output_is_rejected_without_modification(self):
        self.output.mkdir()
        marker = self.output / "keep-me.txt"
        marker.write_bytes(b"pre-existing incomplete output")

        with self.assertRaisesRegex(provision_collision.ProvisionError, "incomplete or contains unregistered"):
            self.provision_with(self.fake_coacd())

        self.assertEqual(list(self.output.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"pre-existing incomplete output")
        self.assertEqual(self.stage_directories(), [])

    def test_existing_corrupted_output_is_rejected_without_modification(self):
        self.provision_with(self.fake_coacd())
        corrupted = self.output / next(iter(self.expected_bytes))
        corrupted.write_bytes(b"preserve corrupted pre-existing bytes")
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}

        with self.assertRaisesRegex(provision_collision.ProvisionError, "SHA-256 mismatch"):
            self.provision_with(self.fake_coacd())

        self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, before)

    def test_orphan_staging_and_unrelated_files_are_never_reused_or_overwritten(self):
        orphan = self.package / ".collision_generated.staging-orphan"
        orphan.mkdir()
        orphan_marker = orphan / "partial.stl"
        orphan_marker.write_bytes(b"orphan from an interrupted process")
        unrelated_before = self.unrelated.read_bytes()

        result = self.provision_with(self.fake_coacd())

        self.assertEqual(result["status"], "GENERATED_AND_HASH_VERIFIED")
        self.assertEqual(orphan_marker.read_bytes(), b"orphan from an interrupted process")
        self.assertEqual(self.unrelated.read_bytes(), unrelated_before)
        self.assert_output_matches_fixture()

    def test_concurrently_appearing_complete_destination_is_reused_without_overwrite(self):
        original_publish = provision_collision._rename_directory_noreplace

        def destination_wins(staging: Path, destination: Path) -> None:
            shutil.copytree(staging, destination)
            original_publish(staging, destination)

        with patch.object(provision_collision, "_rename_directory_noreplace", side_effect=destination_wins):
            result = self.provision_with(self.fake_coacd())

        self.assertEqual(result["status"], "REUSED_HASH_VERIFIED")
        self.assert_output_matches_fixture()
        self.assertEqual(self.stage_directories(), [])
