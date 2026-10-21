"""Offline regressions for the real ROS adapter wait path, without importing ROS."""
import ast
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
import threading

SOURCE = Path(__file__).with_name("issue80_m1_locomotion_probe.py")


def load_function(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
            return namespace[name]
    raise AssertionError(name)


class RosReceiveWaitTests(unittest.TestCase):
    def test_stop_wait_services_all_streams_without_extending_deadline(self):
        now = [0.0]
        received = {k: -1.0 for k in ("odom", "imu", "common", "leg_state", "leg_command")}
        count = [0]
        def spin_once(node, timeout_sec):
            now[0] += max(0.001, timeout_sec)
            received[tuple(received)[count[0] % 5]] = now[0]
            count[0] += 1
        polls = []
        node = SimpleNamespace(_poll_async_readbacks=lambda: polls.append(now[0]),
                               ros_spin_lock=threading.RLock())
        wait = load_function("_spin_for", {
            "time": SimpleNamespace(monotonic=lambda: now[0]),
            "rclpy": SimpleNamespace(ok=lambda: True, spin_once=spin_once),
        })
        wait(node, 1.02)
        self.assertLessEqual(now[0], 1.0211)
        self.assertGreater(count[0], 500)
        self.assertTrue(all(now[0] - value < 0.01 for value in received.values()))
        self.assertEqual(len(polls), count[0])

    def test_result_mapping_is_json_object_not_python_repr_string(self):
        safe = load_function("_json_safe", {"Any": Any, "Mapping": Mapping})
        actual = safe({"activation": {"status": "PASS"}, "samples": [1, 2]})
        self.assertIsInstance(actual, dict)
        self.assertEqual(actual["activation"]["status"], "PASS")
