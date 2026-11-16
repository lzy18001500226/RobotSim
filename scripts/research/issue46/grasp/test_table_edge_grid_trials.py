import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.research.issue46.grasp import table_edge_grid_trials as batch

def complete_source_record():
    return {"gates":{key:"PASS" for key in batch.REQUIRED_GATES},
            "physical_lift_evidence":{"pass":True,"minimum_lift_m":.059,
                "mean_pad_vertical_force_n":1.962,"max_table_force_n":0.,
                "minimum_pad_normal_force_n":{"R_hand_narrow3_Link":14.,"R_hand_wide3_Link":14.}},
            "table_placement":{"pass_result":True,"measured_hold_sim_s":2.02,
                "minimum_table_vertical_force_n":1.961,"maximum_pad_normal_force_n":0.,
                "elapsed_wall_s":20.},
            "task_grasp_region":{"adjustments":1,"elapsed_wall_s":8.},
            "failure":None}

class GridBatchTests(unittest.TestCase):
    def test_pilots_precede_remaining_unique_grid(self):
        self.assertEqual(batch.ORDER[:3],(5,0,10))
        self.assertEqual(sorted(batch.ORDER),list(range(11)))

    def test_initial_offset_is_only_scenario_input(self):
        point={"id":10,"offset_xy_m":[0.,.6459391915312337]}
        argv=batch.command_for(point)
        self.assertIn("--task-grasp-region",argv)
        self.assertEqual(argv[argv.index("--task-bottle-offset-xy-m")+1:][:2],
                         ["0.0","0.6459391915312337"])
        self.assertNotIn("--dock-world-xy",argv)
        self.assertNotIn("--goal",argv)
        self.assertIn("--no-viewer",argv)
        self.assertNotIn("--no-viewer",batch.command_for(point,visible=True))

    def test_existing_gate_and_measured_fields_pass_aggregation(self):
        row=batch.summarize_case(complete_source_record())
        self.assertEqual(row["status"],"PASS")
        self.assertEqual(row["measurements"]["minimum_lift_mm"],59.)
        self.assertEqual(row["measurements"]["maximum_table_support_during_lift_n"],0.)
        self.assertEqual(row["measurements"]["stance_adjustments"],1)
        self.assertFalse(row["new_acceptance_thresholds"])

    def test_old_successful_fixed_dock_cannot_qualify_new_task(self):
        goal={"xy_m":[.821,-1.391],"yaw_rad":0.}
        base=["--execute-pickup-navigation","--task-grasp-region",
              "--dock-world-xy",".821","-1.391","--dock-yaw-rad","0."]
        self.assertTrue(batch.verify_task_entrypoint(base,goal)["pass"])
        wrong=base.copy();wrong[wrong.index("--dock-world-xy")+1]=".951"
        with self.assertRaisesRegex(ValueError,"table-center"):batch.verify_task_entrypoint(wrong,goal)
        missing=base.copy();missing.remove("--task-grasp-region")
        with self.assertRaisesRegex(ValueError,"entrypoint"):batch.verify_task_entrypoint(missing,goal)

    def test_missing_measurement_is_unknown_never_zero_or_pass(self):
        source=complete_source_record()
        del source["physical_lift_evidence"]["minimum_lift_m"]
        row=batch.summarize_case(source)
        self.assertEqual(row["status"],"BLOCKED")
        self.assertIsNone(row["measurements"]["minimum_lift_mm"])
        self.assertIn("minimum_lift_mm",row["missing_evidence"])

    def test_static_or_partial_gate_can_never_finish_batch(self):
        for state in ("NOT_RUN","STATIC_WITNESS"):
            source=complete_source_record()
            source["gates"]["PHYSICAL_TABLE_PLACEMENT"]=state
            self.assertEqual(batch.summarize_case(source)["status"],"BLOCKED")
        source=complete_source_record()
        source["gates"]["HAL_PHYSICAL_LIFT"]="FAIL"
        self.assertEqual(batch.summarize_case(source)["status"],"FAIL")

    def test_nonfinite_and_malformed_metrics_do_not_become_pass(self):
        for value in (float("nan"),"59mm",True,None):
            source=complete_source_record()
            source["physical_lift_evidence"]["minimum_lift_m"]=value
            self.assertEqual(batch.summarize_case(source)["status"],"BLOCKED")

    def test_source_change_stops_before_any_docker_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=root/"batch";task.mkdir()
            (task/"manifest.json").write_text(json.dumps({"source":{"head":"old"}}))
            with patch.object(batch,"REVIEW_ROOT",root),patch.object(batch.pg,"detect_environment",return_value="wsl"),patch.object(batch,"source_identity",return_value={"head":"changed"}),patch.object(batch.subprocess,"run") as run:
                with self.assertRaisesRegex(RuntimeError,"FROZEN_BATCH_SOURCE_CHANGED"):
                    batch.execute(task)
                run.assert_not_called()

    def test_asset_hashes_bind_actual_bytes_and_includes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);asset=root/"shape.stl";asset.write_bytes(b"original")
            scene=root/"scene.xml";scene.write_text('<mujoco><asset><mesh file="shape.stl"/></asset></mujoco>')
            first=batch.scene_assets(scene);asset.write_bytes(b"changed")
            self.assertNotEqual(first,batch.scene_assets(scene))
            scene.write_text('<mujoco><include file="unresolved.xml"/></mujoco>')
            with self.assertRaisesRegex(ValueError,"flattened"):batch.scene_assets(scene)

if __name__=="__main__": unittest.main()
