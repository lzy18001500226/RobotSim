"""Offline tests for SDK/archive interface comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from issue80_compare_aimdk_interfaces import compare_interfaces


class AimdkInterfaceComparisonTests(unittest.TestCase):
    def test_matches_source_interfaces_and_reports_distinct_sdk_source_names(self) -> None:
        package_xml = b"<package><name>aimdk_msgs</name><version>0.0.0</version></package>\n"
        service = b"RequestHeader header\nstring source\nMcActionCommand command\n---\nCommonResponse response\n"
        sdk_root = "aimdk-aarch64-faf4c945-artifacts/src/aimdk_msgs/"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "aimdk-sdk-v1_1_0-aimdk-aarch64-faf4c945-artifacts.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(sdk_root + "package.xml", package_xml)
                archive.writestr(sdk_root + "interface/robot/mc/action/srv/SetMcAction.srv", service)
                archive.writestr(
                    "aimdk-aarch64-faf4c945-artifacts/src/py_examples/py_examples/set_mc_action.py",
                    "req.source = 'node.set_mc_action'\ncmd.action_desc = action_name\n",
                )
                archive.writestr(
                    "aimdk-aarch64-faf4c945-artifacts/src/py_examples/py_examples/set_mc_input_source.py",
                    "req.input_source.name = 'node.set_mc_input_source'\nreq.input_source.priority = 40\n",
                )

            runtime = root / "runtime" / "aimdk_msgs"
            interface = runtime / "interface/robot/mc/action/srv"
            interface.mkdir(parents=True)
            (runtime / "package.xml").write_bytes(package_xml)
            (interface / "SetMcAction.srv").write_bytes(service)
            runtime_example = root / "set_mode.py"
            runtime_example.write_text('source = "rc"\nreq.source = source\n', encoding="utf-8")

            result = compare_interfaces(archive_path, runtime, runtime_example)
            self.assertTrue(result["all_source_interfaces_match"])
            self.assertEqual(result["interface_file_count"], 1)
            self.assertEqual(result["sdk_examples"]["set_mc_action"]["source"], ["req.source = 'node.set_mc_action'"])
            self.assertIn("node.set_mc_input_source", result["sdk_examples"]["set_mc_input_source"]["fields"][0])
            self.assertTrue(any('source = "rc"' in line for line in result["runtime_set_mode_example_source_lines"]))

            (interface / "SetMcAction.srv").write_text("changed schema\n", encoding="utf-8")
            mismatch = compare_interfaces(archive_path, runtime, runtime_example)
            self.assertFalse(mismatch["all_source_interfaces_match"])
            self.assertEqual(mismatch["mismatched_runtime_files"], ["interface/robot/mc/action/srv/SetMcAction.srv"])


if __name__ == "__main__":
    unittest.main()
