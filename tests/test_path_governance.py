from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.agent import path_governance as paths


class PathGovernanceTests(unittest.TestCase):
    def test_detects_windows_wsl_and_native_linux(self) -> None:
        self.assertEqual("windows", paths.detect_environment(platform_name="nt", environ={}))
        self.assertEqual("wsl", paths.detect_environment(platform_name="linux", environ={"WSL_DISTRO_NAME": "Ubuntu-22.04"}))
        self.assertEqual("wsl", paths.detect_environment(platform_name="linux", environ={}, os_release="5.15.90-microsoft-standard-WSL2"))
        self.assertEqual("linux", paths.detect_environment(platform_name="linux", environ={}, os_release="6.8.0-generic"))

    def test_canonicalizes_posix_and_windows_unc_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reviews").mkdir()
            expected = str(root / "reviews" / "issue-43" / "run one" / "result.json")
            expected = expected.replace("\\", "/")
            unc = r"\\wsl.localhost\Ubuntu-22.04" + expected.replace("/", "\\")
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                self.assertEqual(expected, paths.normalize_output_path(expected, environment="wsl"))
                self.assertEqual(expected, paths.normalize_output_path(unc, environment="windows"))
                descriptor = paths.describe_output_path(expected, environment="wsl")
        self.assertEqual(expected, descriptor["WSL_PATH"])
        self.assertEqual(r"\\wsl.localhost\Ubuntu-22.04" + expected.replace("/", "\\"), descriptor["WINDOWS_UNC"])
        self.assertIn("Start-Process -FilePath 'explorer.exe'", descriptor["OPEN_COMMAND"])
        self.assertIn("run one", descriptor["OPEN_COMMAND"])

    def test_rejects_linux_paths_in_native_windows(self) -> None:
        for candidate in ("/home/user/project/reviews/run", "/mnt/c/Users/user/reviews/run"):
            with self.subTest(candidate=candidate), self.assertRaises(paths.PathGovernanceError):
                paths.normalize_output_path(candidate, environment="windows")

    @unittest.skipUnless(
        os.name == "nt" and os.getcwd().casefold().startswith("c:\\"),
        "requires native Windows Python launched from a C: working directory",
    )
    def test_windows_pathlib_cwd_conversion_is_rejected_before_output_creation(self) -> None:
        candidate = (
            Path("/home/lzy18001500226/projects/RobotSim/reviews")
            / f"path-gate-invalid-{uuid.uuid4().hex}"
        ).absolute()
        self.assertTrue(str(candidate).casefold().startswith("c:\\home\\"))
        with self.assertRaises(paths.PathGovernanceError):
            paths.create_output_directory(candidate, environment="windows")
        self.assertFalse(candidate.exists())

    def test_rejects_wrong_distro_and_traversing_unc_paths(self) -> None:
        candidates = (
            r"\\wsl.localhost\Ubuntu-20.04\home\user\projects\RobotSim\reviews\run",
            r"\\wsl.localhost\Ubuntu-22.04\home\user\projects\RobotSim\reviews\..\outside",
            r"\\server\share\RobotSim\reviews\run",
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(paths.PathGovernanceError):
                paths.normalize_output_path(candidate, environment="windows")

    def test_rejects_drive_paths_and_mistyped_windows_homes(self) -> None:
        candidates = (
            r"C:\home\lzy18001500226\projects\RobotSim\reviews\run",
            r"C:\mnt\c\Users\HP\RobotSim\reviews\run",
            r"C:\Users\HP\Desktop\Robot\reviews\run",
            r"C:relative\reviews\run",
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(paths.PathGovernanceError):
                paths.normalize_output_path(candidate, environment="windows")

    def test_rejects_windows_paths_in_wsl_and_mnt_c_outputs(self) -> None:
        for candidate in (r"C:\home\user\reviews\run", "/mnt/c/Users/HP/RobotSim/reviews/run"):
            with self.subTest(candidate=candidate), self.assertRaises(paths.PathGovernanceError):
                paths.normalize_output_path(candidate, environment="wsl")

    def test_rejects_traversal_and_outputs_outside_reviews_without_authorization(self) -> None:
        with self.assertRaises(paths.PathGovernanceError):
            paths.normalize_output_path(paths.CANONICAL_REPO_ROOT + "/reviews/../tmp/run", environment="wsl")
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "capture"
            with self.assertRaises(paths.PathGovernanceError):
                paths.validate_output_path(outside, environment="wsl")

    def test_explicitly_authorized_external_root_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "approved"
            external.mkdir()
            target = external / "capture.json"
            location = paths.validate_output_path(
                target, authorized_roots=[external], environment="wsl"
            )
            self.assertEqual(str(target), location.wsl_path)

    def test_native_linux_output_fails_without_verified_wsl_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reviews").mkdir()
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                with self.assertRaisesRegex(paths.PathGovernanceError, "no verified Ubuntu-22.04 WSL UNC mapping"):
                    paths.validate_output_path(root / "reviews" / "result.json", environment="linux")

    def test_refuses_symlink_escape_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            reviews = root / "reviews"
            reviews.mkdir()
            escape = reviews / "escape"
            escape.symlink_to(Path(outside), target_is_directory=True)
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                target = reviews / "escape" / "should-not-exist"
                with self.assertRaises(paths.PathGovernanceError):
                    paths.create_output_directory(target, environment="wsl")
                self.assertFalse((Path(outside) / "should-not-exist").exists())

    @unittest.skipUnless(os.name == "posix", "uses a POSIX temporary path for WSL policy validation")
    def test_rejects_junction_approved_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = root / "reviews"
            reviews.mkdir()
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                with patch.object(type(reviews), "is_junction", return_value=True, create=True):
                    with self.assertRaisesRegex(paths.PathGovernanceError, "symlink or junction"):
                        paths.validate_output_path(reviews / "run", environment="wsl")

    def test_descriptor_reads_legacy_lowercase_names_but_serializes_canonical_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = root / "reviews"
            run = reviews / "issue-43" / "run"
            run.mkdir(parents=True)
            target = run / "result.json"
            target.write_text("{}\n", encoding="utf-8")
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                descriptor = paths.describe_output_path(target, environment="wsl")
                verified = paths.verify_artifacts([descriptor], environment="wsl")
        self.assertEqual({"WSL_PATH", "WINDOWS_UNC", "OPEN_COMMAND"}, set(descriptor))
        self.assertEqual(descriptor["WINDOWS_UNC"], descriptor["windows_unc"])
        self.assertEqual(descriptor["WINDOWS_UNC"], descriptor.get("windows_unc"))
        self.assertEqual([dict(descriptor)], verified)

    def test_broad_authorization_does_not_expand_the_reviews_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = root / "reviews"
            outside = root / "outside"
            reviews.mkdir()
            outside.mkdir()
            (reviews / "escape").symlink_to(outside, target_is_directory=True)
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                target = reviews / "escape" / "not-approved-by-reviews.json"
                with self.assertRaises(paths.PathGovernanceError):
                    paths.validate_output_path(
                        target, authorized_roots=[root], environment="wsl"
                    )

    def test_unique_output_directory_and_exclusive_artifact_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = root / "reviews"
            experiment = reviews / "issue-43"
            experiment.mkdir(parents=True)
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                first = paths.create_unique_output_directory(
                    experiment,
                    prefix="baseline",
                    environment="wsl",
                    now=datetime(2026, 10, 16, tzinfo=timezone.utc),
                )
                second = paths.create_unique_output_directory(
                    experiment,
                    prefix="baseline",
                    environment="wsl",
                    now=datetime(2026, 10, 16, tzinfo=timezone.utc),
                )
                self.assertNotEqual(first.wsl_path, second.wsl_path)
                file_path = Path(first.wsl_path) / "result.json"
                record = paths.write_new_text(file_path, "{}\n", environment="wsl")
                self.assertTrue(file_path.is_file())
                with self.assertRaises(paths.PathGovernanceError):
                    paths.write_new_text(file_path, "replacement\n", environment="wsl")
                self.assertEqual("{}\n", file_path.read_text(encoding="utf-8"))
                self.assertEqual([record.as_dict()], paths.verify_artifacts([record.as_dict()], environment="wsl"))

    def test_windows_unique_directory_uses_unc_after_internal_wsl_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = root / "reviews"
            reviews.mkdir()
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)), patch.object(
                paths, "_native_path", side_effect=lambda value, _environment: Path(value)
            ):
                parent_unc = paths._windows_unc_for_wsl_path(str(reviews))
                run = paths.create_unique_output_directory(
                    parent_unc,
                    prefix="windows-run",
                    environment="windows",
                    now=datetime(2026, 10, 16, tzinfo=timezone.utc),
                )
                result_unc = paths._windows_unc_for_wsl_path(run.wsl_path + "/result.json")
                record = paths.write_new_text(result_unc, "windows mapped write\n", environment="windows")
                verified = paths.verify_artifacts([record.as_dict()], environment="windows")
                mismatched = record.as_dict()
                mismatched["WSL_PATH"] += "/other-file"
                with self.assertRaises(paths.PathGovernanceError):
                    paths.verify_artifacts([mismatched], environment="windows")
            self.assertEqual(str(reviews), str(Path(run.wsl_path).parent))
            self.assertTrue(run.native_path.is_dir())
            self.assertEqual("windows mapped write\n", (run.native_path / "result.json").read_text())
            self.assertEqual([record.as_dict()], verified)

    def test_missing_or_forged_artifact_mapping_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = root / "reviews"
            run = reviews / "issue-43" / "run"
            run.mkdir(parents=True)
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                target = run / "missing.json"
                descriptor = paths.describe_output_path(target, environment="wsl")
                with self.assertRaises(paths.PathGovernanceError):
                    paths.verify_artifacts([descriptor], environment="wsl")
                target.write_text("{}\n", encoding="utf-8")
                forged = dict(paths.describe_output_path(target, environment="wsl"))
                forged["OPEN_COMMAND"] += "; Remove-Item *"
                with self.assertRaises(paths.PathGovernanceError):
                    paths.verify_artifacts([forged], environment="wsl")

    def test_run_manifest_lists_and_verifies_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "reviews" / "issue-43" / "run"
            run.parent.mkdir(parents=True)
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                paths.create_output_directory(run, environment="wsl")
                (run / "result.json").write_text("{}\n", encoding="utf-8")
                manifest = paths.write_artifact_manifest(run, environment="wsl")
                records = manifest["artifacts"]
                self.assertEqual(2, len(records))
                self.assertTrue(any(item["WSL_PATH"].endswith("result.json") for item in records))
                self.assertTrue(any(item["WSL_PATH"].endswith("artifact_paths.json") for item in records))
                parsed = json.loads((run / "artifact_paths.json").read_text(encoding="utf-8"))
                self.assertEqual(records, parsed["artifacts"])
                paths.verify_artifacts([manifest["manifest"]], environment="wsl")
                (run / "unlisted.txt").write_text("unlisted\n", encoding="utf-8")
                with self.assertRaisesRegex(paths.PathGovernanceError, "does not match"):
                    paths.verify_artifacts([manifest["manifest"]], environment="wsl")
                (run / "unlisted.txt").unlink()
                (run / "result.json").unlink()
                with self.assertRaises(paths.PathGovernanceError):
                    paths.verify_artifacts([manifest["manifest"]], environment="wsl")

    def test_reserved_run_allows_atomic_progress_update_but_not_prior_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "reviews" / "issue-43"
            parent.mkdir(parents=True)
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                run = Path(paths.create_output_directory(parent / "run", environment="wsl").wsl_path)
                result = run / "result.json"
                paths.write_reserved_text(run, result, '{"status":"RUNNING"}\n', environment="wsl")
                paths.write_reserved_text(
                    run, result, '{"status":"FAIL"}\n', overwrite_existing=True, environment="wsl"
                )
                self.assertEqual('{"status":"FAIL"}\n', result.read_text(encoding="utf-8"))
                previous = parent / "previous.json"
                previous.write_text("preserve\n", encoding="utf-8")
                with self.assertRaises(paths.PathGovernanceError):
                    paths.write_reserved_text(run, previous, "overwrite\n", overwrite_existing=True, environment="wsl")
                self.assertEqual("preserve\n", previous.read_text(encoding="utf-8"))

    def test_manifest_requires_a_reserved_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "reviews" / "issue-43" / "run"
            run.mkdir(parents=True)
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                with self.assertRaises(paths.PathGovernanceError):
                    paths.write_artifact_manifest(run, environment="wsl")

    def test_windows_command_escapes_single_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reviews").mkdir()
            target = str(root / "reviews" / "issue-43" / "run's" / "result.json")
            target = target.replace("\\", "/")
            with patch.object(paths, "CANONICAL_REPO_ROOT", str(root)):
                descriptor = paths.describe_output_path(target, environment="wsl")
        self.assertIn("run''s", descriptor["OPEN_COMMAND"])


if __name__ == "__main__":
    unittest.main()
