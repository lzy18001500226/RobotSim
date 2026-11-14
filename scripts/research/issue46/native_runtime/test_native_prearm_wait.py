import ast,unittest,types
from unittest.mock import Mock
from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import prearm_script

class SinglePrearmRequestWaitTests(unittest.TestCase):
    def generated_call(self,wait,needed):
        tree=ast.parse(prearm_script(wait));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='call')
        state={'ready':False};reply=object();future=types.SimpleNamespace(done=lambda:state['ready'],result=lambda:reply)
        client=types.SimpleNamespace(wait_for_service=Mock(return_value=True),call_async=Mock(return_value=future))
        waits=[]
        def spin(node,future,timeout_sec):waits.append(timeout_sec);state['ready']=timeout_sec>=needed
        ns=dict(rclpy=types.SimpleNamespace(spin_until_future_complete=spin),time=types.SimpleNamespace(sleep=lambda _:None))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'generated_prearm','exec'),ns)
        return ns['call'],client,waits,reply
    def test_one_request_can_wait_for_slow_startup_response(self):
        call,client,waits,reply=self.generated_call(2.,.75)
        self.assertIs(call(None,client,object(),attempts=1),reply)
        client.call_async.assert_called_once();self.assertEqual(waits,[2.])
    def test_unanswered_request_still_times_out_without_resubmission(self):
        call,client,waits,_=self.generated_call(2.,3.)
        with self.assertRaisesRegex(RuntimeError,'1 request'):
            call(None,client,object(),attempts=1)
        client.call_async.assert_called_once();self.assertEqual(waits,[2.])
    def test_default_short_wait_remains_unchanged_for_other_callers(self):
        call,client,waits,_=self.generated_call(.25,.75)
        with self.assertRaises(RuntimeError):call(None,client,object(),attempts=1)
        self.assertEqual(waits,[.25]);client.call_async.assert_called_once()
