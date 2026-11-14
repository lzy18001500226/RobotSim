"""Focused pure-XML regression checks for the FoundationWalk STL wrapper."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "native_runtime/foundationwalk_stl_scene_wrapper.py"
SPEC = importlib.util.spec_from_file_location("foundationwalk_stl_scene_wrapper", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FoundationWalkSTLSceneWrapperTests(unittest.TestCase):
    def test_rebinds_only_the_three_named_submodels(self) -> None:
        source = """<mujoco><asset>
          <model name='x2_mc_runtime' file='x2.xml'/>
          <model name='raicom_room' file='room.xml'/>
          <model name='raicom_mug' file='mug.xml'/>
        </asset><worldbody><geom name='floor' type='plane' size='0 0 0.1'/></worldbody></mujoco>"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.xml"
            path.write_text(source, encoding="utf-8")
            tree = MODULE.rebind_wrapper(
                path,
                x2_path="/host/x2_stl.xml",
                room_path="/host/room.xml",
                mug_path="/host/mug.xml",
            )
        root = tree.getroot()
        self.assertEqual(MODULE.model_asset(root, "x2_mc_runtime").get("file"), "/host/x2_stl.xml")
        self.assertEqual(MODULE.model_asset(root, "raicom_room").get("file"), "/host/room.xml")
        self.assertEqual(MODULE.model_asset(root, "raicom_mug").get("file"), "/host/mug.xml")
        self.assertEqual(root.find("./worldbody/geom").get("name"), "floor")

    def test_missing_submodel_is_rejected(self) -> None:
        import xml.etree.ElementTree as ET

        with self.assertRaisesRegex(ValueError, "x2_mc_runtime"):
            MODULE.model_asset(ET.fromstring("<mujoco><asset/></mujoco>"), "x2_mc_runtime")

    def test_runtime_config_repoints_only_the_scene_entry(self) -> None:
        source = """MujocoSimModule:
  simulation_model: /workspace/evidence/old_scene.xml
  robot_model: ${SIM_ROBOT_PATH}/model_info/default.yaml
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "simulator.yaml"
            original.write_text(source, encoding="utf-8")
            rewritten = MODULE.runtime_simulator_config_text(original)
        self.assertIn("simulation_model: /workspace/evidence/scene_foundationwalk_stl_runtime.xml", rewritten)
        self.assertIn("robot_model: ${SIM_ROBOT_PATH}/model_info/default.yaml", rewritten)

    def test_known_mesh_paths_map_to_readonly_container_roots(self) -> None:
        runtime_mesh = MODULE.RAICOM_RUNTIME_ROOT / "sim_mujoco/mesh.stl"
        source_mesh = MODULE.AGIBOT_SOURCE_ROOT / "X2_URDF-v1.4.0/meshes/base_link.stl"
        self.assertEqual(
            MODULE.runtime_mesh_path(runtime_mesh),
            "/workspace/raicom2026/sim_mujoco/mesh.stl",
        )
        self.assertEqual(
            MODULE.runtime_mesh_path(source_mesh),
            "/workspace/agibot_x2_source/X2_URDF-v1.4.0/meshes/base_link.stl",
        )


if __name__ == "__main__":
    unittest.main()
