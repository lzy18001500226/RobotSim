#!/usr/bin/env python3
"""Focused tests for the bounded native STL-hand startup runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("issue46_raicom_stl_native_hand_startup.py")
SPEC = importlib.util.spec_from_file_location("raicom_stl_native_hand_startup", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeHandStartupTests(unittest.TestCase):
    def test_launch_script_uses_only_evidence_local_overlay_config(self) -> None:
        script = MODULE.build_launch_script("simulator_stl_native_hand.yaml")
        self.assertIn("--cfg_file_path=/workspace/evidence/simulator_stl_native_hand.yaml", script)
        self.assertIn("source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash", script)
        self.assertNotIn("/home/", script)
        self.assertNotIn("set -euo", script)

    def test_hand_probe_uses_hal_messages_not_mujoco_state_writes(self) -> None:
        script = MODULE.build_hand_probe_script(-0.6, 1)
        self.assertIn("HandCommandArray", script)
        self.assertIn("/aima/hal/joint/hand/command", script)
        self.assertIn("/aima/hal/joint/hand/state", script)
        self.assertNotIn("qpos", script)
        self.assertNotIn("mj_step", script)
        self.assertIn("node.send_both(-0.59999999999999998)", script)
        self.assertIn("HandType(value=1)", script)

    def test_container_name_is_unique_output_derived(self) -> None:
        output = Path("/tmp/raicom-stl-native-hand-startup-20261030T123456Z")
        self.assertEqual(MODULE.docker_container_name(output), "robotsim-i46-stl-hand-20261030t123456z")


if __name__ == "__main__":
    unittest.main()
