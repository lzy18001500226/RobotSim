#!/usr/bin/env python3
"""Focused tests for the bounded native STL-hand startup runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "native_runtime/stl_hand_startup.py"
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

    def test_mc_launch_uses_only_delivered_runtime_and_evidence_logs(self) -> None:
        script = MODULE.build_mc_launch_script()
        self.assertIn("/workspace/raicom2026/mc/bin", script)
        self.assertIn("LOG_PATH=/workspace/evidence/mc_logs", script)
        self.assertNotIn("set -euo", script)
        self.assertNotIn("/home/", script)

    def test_mc_writeback_paths_are_limited_to_the_known_generated_inputs(self) -> None:
        self.assertEqual(str(MODULE.MC_DUMP_RELATIVE), "mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml.dump")
        self.assertEqual(str(MODULE.MC_GENERATED_RELATIVE), "mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim")

    def test_stand_client_uses_pinned_aimdk_service_contract(self) -> None:
        script = MODULE.build_stand_client_script()
        self.assertIn("SetMcAction", script)
        self.assertIn("GetMcAction", script)
        self.assertIn("/aimdk_5Fmsgs/srv/SetMcAction", script)
        self.assertIn("STAND_DEFAULT", script)

    def test_upper_body_probe_is_bounded_and_restores_the_test_joint(self) -> None:
        script = MODULE.build_upper_body_probe_script()
        self.assertIn("/mc/upper_body_command", script)
        self.assertIn("goal[7] -= 0.05", script)
        self.assertIn("node.publish_for(start, 2.0)", script)
        self.assertNotIn("qpos", script)

    def test_upperbody_split_reuses_the_bounded_vendor_action_contract(self) -> None:
        script = MODULE.build_upperbody_split_client_script()
        self.assertIn("UPPERBODY_REMOTE_SPLIT", script)
        self.assertIn("SetMcAction", script)
        self.assertIn("GetMcAction", script)

    def test_hal_arm_probe_clones_then_restores_the_mc_command(self) -> None:
        script = MODULE.build_hal_arm_probe_script()
        self.assertIn("/aima/hal/joint/arm/command", script)
        self.assertIn("target.position=float(source.position)+0.03", script)
        self.assertIn("node.publish(baseline,2.0)", script)
        self.assertNotIn("qpos", script)

    def test_arm_split_overlay_changes_only_the_mc_arm_publisher_enable_flag(self) -> None:
        source = {"MotionControlModule": {"PublisherManager": {"publisher_cfg_list": [{"name": "leg_joint_command", "enable": True}, {"name": "arm_joint_command", "enable": True}]}}}
        result = MODULE.build_arm_split_mc_config(source)
        self.assertTrue(source["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"][1]["enable"])
        publishers = result["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"]
        self.assertTrue(publishers[0]["enable"])
        self.assertFalse(publishers[1]["enable"])


if __name__ == "__main__":
    unittest.main()
