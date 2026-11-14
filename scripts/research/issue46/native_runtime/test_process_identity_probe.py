"""MC identity discovery preserves all matches without reading unrelated argv."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from scripts.research.issue46.native_runtime.native_base_micro_probe import discover_mc_processes


class ProcessIdentityTests(unittest.TestCase):
    def add(self, root, pid, executable, command):
        directory = root/str(pid)
        directory.mkdir()
        (directory/"exe").symlink_to(executable)
        (directory/"cmdline").write_bytes(command)

    def test_unrelated_process_arguments_are_never_read(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.add(root, 10, "/usr/bin/python3", b"unrelated")
            self.add(root, 20, "/workspace/raicom2026/mc/bin/aima-mc-app-main", b"mc\0--cfg_file_path=verified.yaml\0")
            original = Path.read_bytes
            def read(path):
                if path.parent.name == "10":
                    raise AssertionError("unrelated argv was read")
                return original(path)
            with patch.object(Path, "read_bytes", read):
                candidates = discover_mc_processes(root)
            self.assertEqual([x[0] for x in candidates], [20])
            self.assertIn("--cfg_file_path=verified.yaml", candidates[0][2])

    def test_every_duplicate_mc_is_preserved_for_outer_rejection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pid in (20, 30):
                self.add(root, pid, "/runtime/aima-mc-app-main", b"mc\0")
            self.assertEqual({x[0] for x in discover_mc_processes(root)}, {20, 30})

    def test_disappearing_mc_is_not_returned_as_verified(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.add(root, 20, "/runtime/aima-mc-app-main", b"mc\0")
            (root/"20"/"cmdline").unlink()
            self.assertEqual(discover_mc_processes(root), [])


if __name__ == "__main__":
    unittest.main()
