"""Focused regression: pinned strategy and early completion delivery."""
import ast
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from scripts.research.issue46.grasp import run_lifecycle as life
from scripts.research.issue46.native_runtime import carry_baseline_78df5bd as baseline
from scripts.research.issue46.native_runtime.native_base_micro_probe import build_carry_route_script

ROOT = Path(__file__).resolve().parents[4]

class CompletionAndBaselineTests(unittest.TestCase):
    def test_real_child_event_precedes_cleanup_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            child_code = ("import time; print('RUN_EVENT '+"
                          + repr(json.dumps({"event":"TASK_EXECUTION_FINISHED"}))
                          + ",flush=True); time.sleep(.8)")
            driver_code=("from pathlib import Path; from scripts.research.issue46.grasp.run_lifecycle import forward_process; "
                         "raise SystemExit(forward_process("+repr([sys.executable,"-u","-c",child_code])
                         +",cwd="+repr(str(ROOT))+",log_path=Path("+repr(str(temp/"child.log"))+")))")
            child=subprocess.Popen([sys.executable,"-B","-u","-c",driver_code],
                                   cwd=ROOT,stdout=subprocess.PIPE,text=True)
            first=child.stdout.readline()
            self.assertTrue(first.startswith(life.PREFIX))
            self.assertIsNone(child.poll(), "must notify before slow cleanup")
            self.assertEqual(child.wait(timeout=5),0)
            child.stdout.close()
            self.assertIn("TASK_EXECUTION_FINISHED",(temp/"child.log").read_text())

    def test_observer_distinguishes_route_from_finished_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);seen=set()
            (root/"native_carry_route_result.json").write_text('{"pass_result":false,"failure":"sample"}')
            life.observe_episode(root,seen,destination=root)
            event=json.loads((root/"carry_route_finished.json").read_text())
            self.assertFalse(event["physics_stopped"])
            self.assertFalse(event["cleanup_complete"])
            self.assertLess(event["detected_delay_ms"],500)
            self.assertNotIn("TASK_EXECUTION_FINISHED",seen)
            life.observe_episode(root,seen,destination=root)
            self.assertEqual(len(seen),1)

    def test_controller_fault_surfaces_before_parent_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);seen=set()
            (root/"native_tracking_target.json").write_text('{"pass":true}')
            life.observe_episode(root,seen,destination=root)
            self.assertFalse(seen)
            (root/"native_tracking_target.json").write_text('{"pass":false,"failure":"IK_NOT_CONVERGED"}')
            life.observe_episode(root,seen,destination=root)
            event=json.loads((root/"controller_failure_recorded.json").read_text())
            self.assertEqual(event["failure"],"IK_NOT_CONVERGED")
            self.assertFalse(event["physics_stopped"])
            self.assertFalse((root/"result.json").exists())

    def test_pinned_profile_and_strategy_with_current_guards(self):
        raw=json.loads((ROOT/"simulation/mujoco/scenes/raicom2026/config/reviewed_carry_step_h220_forward_no_roll.json").read_text())
        profile=baseline.task_route_profile(raw,fast_task_timing=True)
        self.assertEqual(profile["delivery_turn_direction"],"CLOCKWISE")
        self.assertEqual(profile["turn_translation_strategy"],"PURE_YAW")
        self.assertEqual(profile["table_front_braking"]["reserve_m"],.75)
        self.assertEqual(profile["retreat_target_m"],.70)
        self.assertEqual(profile["delivery_align_reentry_rad"],.75)
        self.assertTrue(profile["planar_speed_warning_only"])
        self.assertNotIn("measured_turn_stop_delta_rad",profile)
        kwargs=dict(carry_pid=1,timeout_s=90,transit_policy_050=True,
                    away_1m=True,straight_retreat_turn=True,compact_turn=True,
                    transit_command_speed=.20,compact_centre_feedback=True,
                    release_source_on_stop=True,compact_angular_cap=.60,
                    delivery=profile,vendor_transition_table=True)
        current=build_carry_route_script(None,{},[],**kwargs)
        restored=build_carry_route_script(None,{},[],baseline_78df5bd=True,**kwargs)
        def functions(text):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        a,b=functions(current),functions(restored)
        changed={k for k in a if a[k]!=b[k]}
        self.assertEqual(changed,{"run_carry_navigation","table_front_braking_observation"})
        for key in ("physical_fault","carry_owner_fault","carry_motion_handshake","planned_step_deceleration"):
            self.assertEqual(a[key],b[key],key)
        self.assertNotIn("delivery_heading_recovery(mission",ast.get_source_segment(restored,next(n for n in ast.parse(restored).body if isinstance(n,ast.FunctionDef) and n.name=="run_carry_navigation")))
        compile(restored,"restored","exec")

if __name__=="__main__":
    unittest.main()
