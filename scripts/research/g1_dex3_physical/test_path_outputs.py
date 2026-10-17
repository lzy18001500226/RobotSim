from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import baseline
from agent import path_governance


class G1Dex3PathCallerTests(unittest.TestCase):
    def test_baseline_plan_and_episode_use_the_shared_run_allocator(self) -> None:
        root = Path(__file__).resolve().parents[3]
        for relative in (
            "scripts/research/g1_dex3_physical/baseline.py",
            "scripts/research/g1_dex3_physical/plan.py",
            "scripts/research/g1_dex3_physical/episode.py",
        ):
            with self.subTest(entrypoint=relative):
                tree = ast.parse((root / relative).read_text(encoding="utf-8"))
                calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "reserve_output_run"
                ]
                self.assertTrue(calls)

    def test_reserved_run_is_unique_and_manifest_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            parent = repo / "reviews" / "issue-43" / "dex3-physical"
            parent.mkdir(parents=True)
            with patch.object(path_governance, "CANONICAL_REPO_ROOT", str(repo)), patch.object(
                path_governance, "detect_environment", return_value="wsl"
            ), patch.object(baseline, "OUT", parent):
                first = baseline.reserve_output_run("baseline-test")
                second = baseline.reserve_output_run("baseline-test")
                self.assertNotEqual(first, second)
                path_governance.write_new_text(first / "audit.json", "{}\n")
                manifest = path_governance.write_artifact_manifest(first)
                verified = path_governance.verify_artifacts([manifest["manifest"]])
                self.assertTrue((first / "artifact_paths.json").is_file())

        self.assertEqual(2, len(manifest["artifacts"]))
        self.assertEqual(1, len(verified))

    def test_invalid_windows_and_wsl_roots_fail_before_native_path_resolution(self) -> None:
        cases = (
            ("windows", r"C:\home\lzy18001500226\projects\RobotSim\reviews\run"),
            ("wsl", "/mnt/c/Users/HP/RobotSim/reviews/run"),
        )
        for runtime, invalid_root in cases:
            with self.subTest(runtime=runtime):
                with patch.object(baseline, "OUT", invalid_root), patch.object(
                    path_governance, "detect_environment", return_value=runtime
                ), patch.object(path_governance, "_native_path") as native_path:
                    with self.assertRaises(path_governance.PathGovernanceError):
                        baseline.reserve_output_run("must-not-write")
                    native_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
