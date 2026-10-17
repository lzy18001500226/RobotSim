from __future__ import annotations

import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import issue46_native_omnihand_geometry_first as geometry
import issue46_native_omnihand_minimal_grasp as fixture


class X2OmniHandPathCallerTests(unittest.TestCase):
    def test_main_and_standalone_fixture_use_reserved_run_allocators(self) -> None:
        root = Path(__file__).resolve().parents[2]
        callers = (
            (
                root / "scripts/research/issue46_native_omnihand_geometry_first.py",
                "reserve_evidence_root",
            ),
            (
                root / "scripts/research/issue46_native_omnihand_minimal_grasp.py",
                "reserve_fixture_output",
            ),
        )
        for source, allocator in callers:
            with self.subTest(caller=source.name):
                tree = ast.parse(source.read_text(encoding="utf-8"))
                calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == allocator
                ]
                self.assertTrue(calls)

    def test_geometry_main_allocates_unique_manifested_output_without_physics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            parent = repo / "reviews" / "issue-46" / "native-omnihand-grasp"
            parent.mkdir(parents=True)

            def analyze(candidate: dict[str, object], output_dir: Path) -> dict[str, object]:
                geometry.path_governance.create_output_directory(output_dir)
                geometry.path_governance.write_new_text(
                    output_dir / "geometry_result.json", "{}\n"
                )
                return {
                    "candidate": candidate["name"],
                    "bilateral_target": {"status": "BLOCKED"},
                    "independent_family_contact_targets": {},
                    "open_pose_metrics": {
                        "family_to_bottle": {
                            family: {"distance_m": 0.01} for family in geometry.FAMILIES
                        }
                    },
                }

            with patch.object(geometry.path_governance, "CANONICAL_REPO_ROOT", str(repo)), patch.object(
                geometry.path_governance, "detect_environment", return_value="wsl"
            ), patch.object(geometry, "EVIDENCE_PARENT", parent), patch.object(
                geometry, "EVIDENCE_ROOT", parent
            ), patch.object(geometry.fixture, "verify_vendor_pin"), patch.object(
                geometry.fixture, "git_output", return_value="575cc6b988f976c23550e0db85aa1e5475d3652d"
            ), patch.object(geometry, "sha256", return_value="0" * 64), patch.object(
                geometry, "analyze_candidate", side_effect=analyze
            ), patch("sys.argv", ["geometry-first", "--candidate", "prior_fixed_palm", "--geometry-only"]):
                with patch.object(geometry, "run_dynamic_candidate") as dynamic, contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    status = geometry.main()
                    dynamic.assert_not_called()
                runs = list(parent.glob("geometry-first-*"))
                self.assertEqual(1, len(runs))
                manifest_path = runs[0] / "artifact_paths.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                verified = geometry.path_governance.verify_artifacts(
                    [geometry.path_governance.describe_output_path(manifest_path)]
                )

        self.assertEqual(0, status)
        self.assertEqual("robotsim.local-artifacts.v1", manifest["schema_version"])
        self.assertEqual(3, len(manifest["artifacts"]))
        self.assertEqual(1, len(verified))

    def test_fixture_standalone_allocator_uses_unique_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            parent = repo / "reviews" / "issue-46" / "native-omnihand-grasp"
            parent.mkdir(parents=True)
            with patch.object(fixture.path_governance, "CANONICAL_REPO_ROOT", str(repo)), patch.object(
                fixture.path_governance, "detect_environment", return_value="wsl"
            ), patch.object(fixture, "OUT", parent):
                first = fixture.reserve_fixture_output("fixture-test")
                second = fixture.reserve_fixture_output("fixture-test")
                self.assertNotEqual(first, second)
                fixture.path_governance.write_new_text(first / "result.json", "{}\n")
                manifest = fixture.path_governance.write_artifact_manifest(first)
                verified = fixture.path_governance.verify_artifacts([manifest["manifest"]])

        self.assertEqual(2, len(manifest["artifacts"]))
        self.assertEqual(1, len(verified))

    def test_invalid_windows_and_wsl_roots_fail_before_native_path_resolution(self) -> None:
        cases = (
            ("windows", r"C:\home\lzy18001500226\projects\RobotSim\reviews\run"),
            ("wsl", "/mnt/c/Users/HP/RobotSim/reviews/run"),
        )
        for runtime, invalid_root in cases:
            with self.subTest(runtime=runtime):
                with patch.object(geometry, "EVIDENCE_PARENT", invalid_root), patch.object(
                    geometry.path_governance, "detect_environment", return_value=runtime
                ), patch.object(geometry.path_governance, "_native_path") as native_path:
                    with self.assertRaises(geometry.path_governance.PathGovernanceError):
                        geometry.reserve_evidence_root("must-not-write")
                    native_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
