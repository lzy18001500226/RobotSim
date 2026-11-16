"""Keep active height zeroing and released-source observation distinct."""
import unittest
from scripts.research.issue46.native_runtime.native_body_height import body_pause_zero_required
class BodyReleaseTests(unittest.TestCase):
    def test_active_publisher_must_receive_zero(self):
        self.assertTrue(body_pause_zero_required(True,False,False))
    def test_verified_release_and_low_stop_do_not_write_dead_pipe(self):
        self.assertFalse(body_pause_zero_required(False,True,True))
    def test_unconfirmed_release_is_not_treated_as_safe_observation(self):
        for disabled,stopped in ((False,True),(True,False),(False,False)):
            with self.assertRaisesRegex(RuntimeError,"BODY_PUBLISHER_NOT_ALIVE"):
                body_pause_zero_required(False,disabled,stopped)
class InitialStopEpochTests(unittest.TestCase):
    def test_clock_change_after_initial_stop_requires_new_full_stop_without_actuation(self):
        import ast,inspect,types
        from scripts.research.issue46.native_runtime import native_body_height as body
        generated=body.build_script({},[])
        tree=ast.parse(generated)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_height_probe")
        fn=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=="sample_after_qualified_stop")
        events=[];result={"initial_stop":{"pass":True},"commands":[]};count=[0]
        def sample():
            count[0]+=1;events.append("sample")
            if count[0]==1:raise RuntimeError("CLOCK_GENERATION_PENDING")
            return {"fresh":True}
        def settle(*args):events.append("new_full_stop");return {"pass":True,"new_window":True}
        ns=dict(sample=sample,result=result,n=types.SimpleNamespace(latest_row={"generation":1,"process_pid":46}),process=None,
            body_pause_zero_required=body.body_pause_zero_required,command=lambda _:self.fail("read-only observer cannot actuate"),
            settle_native_body=settle,deadline=10.,time=types.SimpleNamespace(monotonic=lambda:1.))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),"actual_generated_stop_helper","exec"),ns)
        self.assertEqual(ns["sample_after_qualified_stop"]("initial_stop"),{"fresh":True})
        self.assertEqual(events,["sample","new_full_stop","sample"])
        self.assertTrue(result["initial_stop"]["new_window"])
        self.assertTrue(result["initial_readonly_clock_recheck"])
        self.assertIn('initial=sample_after_qualified_stop("initial_stop")',generated)
if __name__=="__main__":unittest.main()
