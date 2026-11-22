from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agent import eval_harness, stage_checkpoint as stage


@unittest.skipUnless(os.name == "posix", "stage export targets WSL source checkouts")
class StageCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="robotsim-stage-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        eval_harness._init_repo(self.repo, {"owned.py": "VALUE = 1\n", "unrelated.txt": "keep\n"})
        self.evidence = self.root / "reviews" / "result.json"
        self.evidence.parent.mkdir()
        self.evidence.write_text('{"acceptance":"synthetic only"}\n')
        self.output = self.root / "reviews"
        self.mapping = patch.object(stage.paths, "detect_environment", return_value="wsl")
        self.mapping.start()
        self.addCleanup(self.mapping.stop)

    def export(self, names=None, deleted=None, **kwargs):
        names = names or ["owned.py"]
        return stage.export(self.repo, stage="synthetic-stage", names=names,
                            evidence=[self.evidence], reproduction="python3 -B owned.py",
                            validated_fingerprint=stage.fingerprint(stage._sources(self.repo, names, deleted)[0]),
                            output_parent=self.output, authorized_roots=(self.root,), deleted=deleted, **kwargs)

    def test_dirty_and_untracked_scope_recovers_without_changing_git_or_unrelated_work(self):
        (self.repo / "owned.py").write_text("VALUE = 2\n")
        (self.repo / "new.py").write_text("NEW = True\n")
        (self.repo / "unrelated.txt").write_text("unrelated dirty work\n")
        before = stage._git(self.repo, "status", "--porcelain=v1")
        head = stage._git(self.repo, "rev-parse", "HEAD")
        packet_path = self.export(["owned.py", "new.py"])
        packet = json.loads(packet_path.read_text())
        self.assertEqual(packet["publication"], "NOT_CHECKED")
        self.assertFalse(packet["task_completed"])
        self.assertIn("producer assertion", packet["acceptance"])
        self.assertEqual((packet_path.parent / "source" / "new.py").read_text(), "NEW = True\n")
        self.assertEqual((packet_path.parent / "source" / "owned.py").read_text(), "VALUE = 2\n")
        self.assertFalse((packet_path.parent / "source" / "unrelated.txt").exists())
        self.assertNotIn("unrelated", (packet_path.parent / "review.patch").read_text())
        self.assertTrue((packet_path.parent / "artifact_paths.json").is_file())
        self.assertEqual(stage._git(self.repo, "status", "--porcelain=v1"), before)
        self.assertEqual(stage._git(self.repo, "rev-parse", "HEAD"), head)

    def test_changed_validated_source_is_rejected_before_output(self):
        old = stage.fingerprint(stage._sources(self.repo, ["owned.py"])[0])
        (self.repo / "owned.py").write_text("unvalidated change\n")
        with self.assertRaisesRegex(ValueError, "validated fingerprint"):
            stage.export(self.repo, stage="synthetic", names=["owned.py"], evidence=[self.evidence],
                         reproduction="python3 owned.py", validated_fingerprint=old,
                         output_parent=self.output, authorized_roots=(self.root,))
        self.assertEqual(list(self.output.iterdir()), [self.evidence])

    def test_private_escape_and_symlink_sources_are_rejected(self):
        for name in ("", ".", "../outside.py", ".git/HEAD", ".env", ".codex/config.toml", "/etc/passwd"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                stage._sources(self.repo, [name])
        (self.repo / "alias.py").symlink_to(self.repo / "owned.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            stage._sources(self.repo, ["alias.py"])
        (self.repo / ".codex").mkdir()
        (self.repo / ".codex" / "hooks.json").write_text('{"hooks":{}}\n')
        self.assertIn(".codex/hooks.json", stage._sources(self.repo, [".codex/hooks.json"])[0])

    def test_published_checkpoint_requires_clean_scoped_source_and_matching_remote(self):
        stage._git(self.repo, "switch", "-c", "codex/stage")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "run.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.repo / "src" / "run.sh").chmod(0o755)
        stage._git(self.repo, "add", "src/run.sh")
        stage._git(self.repo, "commit", "-m", "nested executable stage input")
        remote = self.root / "remote.git"
        stage._git(self.repo, "init", "--bare", str(remote))
        stage._git(self.repo, "remote", "add", "origin", str(remote))
        with self.assertRaises(ValueError):
            self.export(verify_published=True)
        stage._git(self.repo, "push", "origin", "codex/stage")
        # Unrelated dirty work does not prevent verifying this committed scope.
        (self.repo / "unrelated.txt").write_text("other owner\n")
        self.assertEqual(json.loads(self.export(["owned.py", "src/run.sh"], verify_published=True).read_text())["publication"], "VERIFIED")
        (self.repo / "owned.py").write_text("dirty again\n")
        with self.assertRaisesRegex(ValueError, "clean explicit scope"):
            self.export(verify_published=True)

    def test_source_changes_during_export_leave_no_completed_packet(self):
        original = stage._sources
        calls = 0
        def changing(repo, names, deleted=None):
            nonlocal calls
            calls += 1
            if calls == 3:  # helper fingerprint, export capture, export recheck
                (repo / "owned.py").write_text("changed while exporting\n")
            return original(repo, names, deleted)
        with patch.object(stage, "_sources", side_effect=changing), self.assertRaisesRegex(ValueError, "changed during export"):
            self.export()
        self.assertFalse(list(self.output.rglob("stage_handoff.json")))

    def test_ignored_untracked_source_cannot_be_reported_published(self):
        stage._git(self.repo, "switch", "-c", "codex/stage")
        (self.repo / ".gitignore").write_text("helper.py\n")
        stage._git(self.repo, "add", ".gitignore")
        stage._git(self.repo, "commit", "-m", "ignore local helper")
        remote = self.root / "remote.git"
        stage._git(self.repo, "init", "--bare", str(remote))
        stage._git(self.repo, "remote", "add", "origin", str(remote))
        stage._git(self.repo, "push", "origin", "codex/stage")
        (self.repo / "helper.py").write_text("not in Git\n")
        with self.assertRaisesRegex(ValueError, "not committed"):
            self.export(["helper.py"], verify_published=True)
        stage._git(self.repo, "config", "core.filemode", "false")
        (self.repo / "owned.py").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "not committed"):
            self.export(verify_published=True)

    def test_deletion_and_rename_retain_the_old_path_disposition(self):
        (self.repo / "owned.py").rename(self.repo / "renamed.py")
        packet_path = self.export(["renamed.py"], deleted=["owned.py"])
        packet = json.loads(packet_path.read_text())
        self.assertEqual(packet["sources"]["owned.py"], {"deleted": True})
        self.assertIn("deleted file", (packet_path.parent / "review.patch").read_text())
        self.assertEqual((packet_path.parent / "source" / "renamed.py").read_bytes(), (self.repo / "renamed.py").read_bytes())
        self.assertFalse((packet_path.parent / "source" / "owned.py").exists())
        stage._git(self.repo, "switch", "-c", "codex/renamed")
        stage._git(self.repo, "add", "--", "owned.py", "renamed.py")
        stage._git(self.repo, "commit", "-m", "rename selected stage source")
        remote = self.root / "remote.git"
        stage._git(self.repo, "init", "--bare", str(remote))
        stage._git(self.repo, "remote", "add", "origin", str(remote))
        stage._git(self.repo, "push", "origin", "codex/renamed")
        published = json.loads(self.export(["renamed.py"], deleted=["owned.py"], verify_published=True).read_text())
        self.assertEqual(published["publication"], "VERIFIED")
        self.assertEqual(published["source_fingerprint"], packet["source_fingerprint"])


if __name__ == "__main__":
    unittest.main()
