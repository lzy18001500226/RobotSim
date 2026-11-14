"""Exercise the generated single-request transport with an elapsed deadline."""
import ast
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import prearm_script

class SinglePrearmRequestWaitTests(unittest.TestCase):
    def generated_call(self,wait,needed):
        tree=ast.parse(prearm_script(wait));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='call_service')
        state={'time':0.,'cancelled':False};reply=object()
        future=NS(done=lambda:state['time']>=needed,result=lambda:reply,
                  cancelled=lambda:state['cancelled'],cancel=lambda:state.update(cancelled=True))
        client=NS(srv_name='/SetMcAction',wait_for_service=Mock(return_value=True),
                  call_async=Mock(return_value=future),remove_pending_request=Mock())
        def spin_once(timeout_sec):state['time']+=timeout_sec
        node=NS(executor=NS(spin_once=spin_once))
        ns=dict(infra_elapsed=lambda:state['time'],ModeHandshakeError=RuntimeError)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'generated_prearm','exec'),ns)
        def invoke():
            with patch.dict(sys.modules,rclpy=NS(ok=lambda:True)):
                return ns['call_service'](node,client,object(),timeout_s=wait)
        return invoke,client,state,reply
    def test_one_request_can_wait_for_slow_startup_response(self):
        call,client,state,reply=self.generated_call(2.,.75)
        self.assertIs(call(),reply)
        client.call_async.assert_called_once();self.assertLess(state['time'],2.)
        self.assertFalse(state['cancelled'])
    def test_unanswered_request_still_times_out_without_resubmission(self):
        call,client,state,_=self.generated_call(2.,3.)
        with self.assertRaisesRegex(RuntimeError,'service timeout:'):call()
        client.call_async.assert_called_once();client.remove_pending_request.assert_called_once()
        self.assertTrue(state['cancelled']);self.assertLessEqual(state['time'],2.000001)
    def test_short_explicit_wait_is_not_silently_extended(self):
        call,client,state,_=self.generated_call(.25,.75)
        with self.assertRaisesRegex(RuntimeError,'service timeout:'):call()
        self.assertLessEqual(state['time'],.250001);client.call_async.assert_called_once()
