from __future__ import annotations

import ast
import unittest
from pathlib import Path


class Local04BalanceHandoffTests(unittest.TestCase):
    def test_reads_native_trace_and_refuses_docking_claim(self) -> None:
        source = Path(__file__).with_name("issue80_local04_balance_handoff.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("PASS_BALANCE_AT_APPROACH_ENDPOINT", source)
        self.assertIn("TABLE_DOCKING and grasp are NOT RUN", source)
        self.assertIn("last_complete_post_step", source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)
