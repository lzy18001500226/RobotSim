"""Actual generated-route input ramp; no gait or physics claims."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
from scripts.research.issue46.native_runtime import native_base_micro_probe as m


class RetreatInputTests(unittest.TestCase):
    def run_input(self, phase, elapsed, rearward=False):
        tree=ast.parse(Path(m.__file__).read_text())
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                   and any(isinstance(a,ast.Assign) and
                           "retreat_input_ramp_s" in ast.unparse(a)
                           and isinstance(a.value,ast.Constant) and a.value.value==2.
                           for a in n.body))
        ns=dict(cfg=dict(transit_command_speed=.20,delivery=dict(rearward_foundation_transit=rearward)),mission=dict(phase=phase),tick=100.+elapsed,activation_started=100.,
                forward=-.20,angular=.10,bounded_start_scale=m.bounded_start_scale)
        exec(compile(ast.Module(body=[block],type_ignores=[]),"actual-route-start","exec"),ns)
        return ns

    def test_held_stand_starts_with_zero_then_reaches_original_command(self):
        start=self.run_input("RETREAT",0.)
        self.assertEqual((start["forward"],start["angular"]),(0.,0.))
        half=self.run_input("RETREAT",1.)
        self.assertAlmostEqual(half["forward"],-.10)
        self.assertAlmostEqual(half["angular"],.05)
        end=self.run_input("RETREAT",2.)
        self.assertEqual((end["forward"],end["angular"]),(-.20,.10))
        self.assertEqual(end["mission"]["retreat_input_ramp_s"],2.)

    def test_first_three_hundred_ms_cannot_reproduce_saved_full_reverse_step(self):
        self.assertLess(abs(self.run_input("RETREAT",.3)["forward"]),.013)
        commands=[self.run_input("RETREAT",i*.02)["forward"] for i in range(151)]
        self.assertTrue(all(-.20<=v<=0. for v in commands))
        self.assertTrue(all(b<=a for a,b in zip(commands,commands[1:])))

    def test_existing_rearward_foundation_ramp_is_not_applied_twice(self):
        value=self.run_input("RETREAT",.1,rearward=True)
        self.assertEqual((value["forward"],value["angular"]),(-.20,.10))
        self.assertNotIn("retreat_input_ramp_s",value["mission"])

    def test_other_mission_phases_are_unchanged(self):
        for phase in ["TURN","FORWARD","EGRESS","DELIVERY"]:
            value=self.run_input(phase,.1)
            self.assertEqual((value["forward"],value["angular"]),(-.20,.10))
            self.assertNotIn("retreat_input_ramp_s",value["mission"])



class DeliveryInputTests(unittest.TestCase):
    def run_input(self, elapsed, ramp=2.):
        tree=ast.parse(Path(m.__file__).read_text())
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                   and isinstance(n.test,ast.BoolOp)
                   and "forward_activation_ramp_s" in ast.unparse(n.test))
        ns=dict(cfg=dict(delivery=dict(forward_activation_ramp_s=ramp)),
                mission=dict(phase="DELIVERY"),output=type("Output",(),dict(state="DRIVE"))(),
                NavState=NS(DRIVE="DRIVE"),tick=100.+elapsed,activation_started=100.,
                forward=.20,bounded_start_scale=m.bounded_start_scale)
        exec(compile(ast.Module(body=[block],type_ignores=[]),"actual-delivery-start","exec"),ns)
        return ns["forward"]
    def test_actual_delivery_command_no_longer_starts_at_full_cruise(self):
        self.assertEqual(self.run_input(0.),0.)
        self.assertLess(self.run_input(.3),.013)
        self.assertAlmostEqual(self.run_input(1.),.10)
        self.assertEqual(self.run_input(2.),.20)
    def test_existing_one_second_linear_profile_is_unchanged(self):
        self.assertAlmostEqual(self.run_input(.25,1.),.05)
    def test_generated_ramp_is_monotonic_within_original_cruise_command(self):
        commands=[self.run_input(i*.02) for i in range(151)]
        self.assertTrue(all(0.<=x<=.20 for x in commands))
        self.assertTrue(all(b>=a for a,b in zip(commands,commands[1:])))

if __name__=="__main__":
    unittest.main()
