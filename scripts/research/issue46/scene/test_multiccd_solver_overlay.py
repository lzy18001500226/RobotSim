import json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
from scripts.research.issue46.scene import multiccd_solver_overlay as m
class MulticcdOverlayTests(unittest.TestCase):
    def test_only_solver_flag_changes_in_both_scene_variants(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);source=folder/"model.xml"
            source.write_text('<mujoco><option timestep=".001"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .2"><freejoint/><geom type="cylinder" size=".035 .15" mass=".2"/></body></worldbody></mujoco>')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            cfg=folder/"simulator.yaml";cfg.write_text(m.yaml.safe_dump({"MujocoSimModule":{"simulation_model":"/workspace/evidence/model.xml"}}))
            parent=folder/"parent.json";parent.write_text(json.dumps(dict(generated={**{k:dict(path=str(source),sha256=sha(source)) for k in ("host_scene","runtime_scene","runtime_hostcheck_scene")},"simulator_config":dict(path=str(cfg),sha256=sha(cfg))})))
            out=folder/"out";out.mkdir()
            with patch.object(m.pg,"create_unique_output_directory",return_value=SimpleNamespace(native_path=out)), patch.object(m.pg,"write_new_text",side_effect=lambda path,text:Path(path).write_text(text)), patch.object(m.pg,"write_artifact_manifest"):
                path=m.build(parent)
            result=json.loads(path.read_text());e=result["numeric_contact_manifold"]
            self.assertEqual((e["old_enableflags"],e["new_enableflags"]),(0,16))
            self.assertGreater(len(e["geometry_material_mass_gain_limit_arrays_unchanged"]),100)
            self.assertIn("runtime_scene.xml",Path(result["generated"]["simulator_config"]["path"]).read_text())
            self.assertIn("multiccd",Path(result["generated"]["runtime_hostcheck_scene"]["path"]).read_text())
            self.assertNotIn("multiccd",source.read_text())
if __name__=="__main__":unittest.main()
