from __future__ import annotations

import ast
import unittest
from pathlib import Path


class GetupAuditSourceTests(unittest.TestCase):
    def test_is_read_only_and_requires_all_transition_edges(self) -> None:
        source = Path(__file__).with_name("issue80_getup_capability_audit.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("PASSIVE_DEFAULT", source)
        self.assertIn("LIE_UP_DEFAULT", source)
        self.assertIn("GROUNDUP", source)
        self.assertIn("STAND_DEFAULT", source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)
        self.assertNotIn("mj_step", calls)
