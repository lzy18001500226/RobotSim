from __future__ import annotations

import ast
import unittest
from pathlib import Path


class PncRuntimeAuditTests(unittest.TestCase):
    def test_audit_is_read_only_and_checks_optional_components(self) -> None:
        source = Path(__file__).with_name("issue80_pnc_runtime_audit.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)
        self.assertIn("PncTaskRequest", source)
        self.assertIn("task_engine", source)
        self.assertIn("BLOCKED_PNC_MODULE_ABSENT_FROM_X86_SIM_DELIVERY", source)


if __name__ == "__main__":
    unittest.main()
