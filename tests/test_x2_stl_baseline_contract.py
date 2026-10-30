"""Prevent the active X2 task configuration from silently selecting old models."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "simulation/mujoco/models/agibot_x2/registry.json"
SCENE_CONFIG = ROOT / "simulation/mujoco/scenes/raicom2026/config/x2_stl_autonomy_scene.json"
RESEARCH_ROOT = ROOT / "scripts/research"


class X2STLBaselineContractTests(unittest.TestCase):
    def test_registry_default_is_the_frozen_stl_model(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(registry["default_model_id"], "X2-OmniPicker-STL")
        self.assertEqual(set(registry["models"]), {"X2-OmniPicker-CAD", "X2-OmniPicker-STL"})

    def test_raicom_scene_config_has_no_disposable_review_or_worktree_dependency(self) -> None:
        raw = SCENE_CONFIG.read_text(encoding="utf-8")
        config = json.loads(raw)
        self.assertEqual(config["schema"], "robotsim.raicom2026.x2-stl-autonomy-scene.v3")
        self.assertNotIn("reviews/", raw)
        self.assertNotIn("robotsim-issue", raw)
        self.assertNotIn("X2-OmniPicker-CAD", raw)

    def test_issue46_tools_are_grouped_under_one_catalog(self) -> None:
        self.assertTrue((RESEARCH_ROOT / "issue46/README.md").is_file())
        self.assertEqual(list(RESEARCH_ROOT.glob("issue46_*.py")), [])
        self.assertEqual(list(RESEARCH_ROOT.glob("test_issue46_*.py")), [])


if __name__ == "__main__":
    unittest.main()
