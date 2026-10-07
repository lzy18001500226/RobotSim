from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.agent.check_governance import (
    SMALL_FIXTURE_MAX_BYTES,
    check_repository,
)


class GovernanceCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self._write(
            "AGENTS.md",
            "[Runtime](docs/engineering/runtime.md#prototype-lifecycle)\n"
            "[Validation](docs/engineering/validation.md#test-retention-and-generated-evidence)\n",
        )
        self._write("docs/engineering/runtime.md", "runtime policy\n")
        self._write("docs/engineering/validation.md", "validation policy\n")
        self._stage("AGENTS.md", "docs/engineering/runtime.md", "docs/engineering/validation.md")

    def _write(self, relative: str, content: str | bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def _stage(self, *paths: str) -> None:
        subprocess.run(["git", "-C", str(self.root), "add", "--", *paths], check=True)

    def test_policy_documents_must_remain_linked_and_present(self) -> None:
        self._write("AGENTS.md", "Only general repository rules.\n")
        self._stage("AGENTS.md")
        problems = check_repository(self.root)
        self.assertTrue(any("AGENTS.md must link to docs/engineering/runtime.md" in item for item in problems))
        self.assertTrue(any("AGENTS.md must link to docs/engineering/validation.md" in item for item in problems))

    def test_direct_runtime_path_and_python_imports_are_rejected(self) -> None:
        self._write("robots/unitree_g1/runtime.cpp", 'run("python scripts/research/try.py");\n')
        self._write("simulation/mujoco/runner.py", "from scripts import research\n")
        self._stage("robots/unitree_g1/runtime.cpp", "simulation/mujoco/runner.py")
        problems = check_repository(self.root)
        self.assertTrue(any("robots/unitree_g1/runtime.cpp" in item for item in problems))
        self.assertTrue(any("simulation/mujoco/runner.py" in item for item in problems))

    def test_generated_outputs_are_blocked_but_small_fixtures_and_source_images_are_allowed(self) -> None:
        self._write("reviews/run/summary.json", "{}\n")
        self._write("scripts/research/run.jsonl", '{"sample":1}\n')
        self._write("apps/unity/Assets/robot_icon.png", b"source image")
        self._write("tests/fixtures/expected.png", b"small fixture")
        self._stage(
            "reviews/run/summary.json",
            "scripts/research/run.jsonl",
            "apps/unity/Assets/robot_icon.png",
            "tests/fixtures/expected.png",
        )
        problems = check_repository(self.root)
        self.assertTrue(any("reviews/run/summary.json" in item for item in problems))
        self.assertTrue(any("scripts/research/run.jsonl" in item for item in problems))
        self.assertFalse(any("robot_icon.png" in item for item in problems))
        self.assertFalse(any("expected.png" in item for item in problems))

    def test_large_json_is_rejected_outside_small_fixture_allowance(self) -> None:
        self._write("data/experiment.json", b"0" * (SMALL_FIXTURE_MAX_BYTES + 1))
        self._stage("data/experiment.json")
        problems = check_repository(self.root)
        self.assertTrue(any("data/experiment.json" in item for item in problems))


if __name__ == "__main__":
    unittest.main()
