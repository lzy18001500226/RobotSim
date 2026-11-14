"""Regression of the generated Dock.stand call site, without robot motion."""
import ast
import unittest
from types import SimpleNamespace as NS
from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as launch


class GeneratedHandshakeTests(unittest.TestCase):
    def test_armed_step_is_explicitly_not_applied_motion(self):
        from scripts.research.issue46.native_runtime.mode_handshake import ensure_mode
        sent=[]
        row=dict(name='STAND_DEFAULT',status=100,header_code=0)
        result=ensure_mode('LOCOMOTION_STEP',lambda _:row,
                           lambda _:sent.append(True) or dict(status=1,header_code=0),
                           pending_readiness=lambda:dict(step_speed_mode=4))
        self.assertEqual(sent,[True])
        self.assertEqual(result['outcome'],'ARMED_PENDING_VELOCITY')
        self.assertIsNone(result['applied'])

    def test_pending_contract_cannot_be_used_for_other_modes(self):
        from scripts.research.issue46.native_runtime.mode_handshake import ensure_mode
        with self.assertRaisesRegex(ValueError,'only defined for LOCOMOTION_STEP'):
            ensure_mode('STAND_DEFAULT',lambda _:None,lambda _:None,pending_readiness=lambda:{})

    def invoke(self, initial="STAND_DEFAULT", *, lose_ack=False, reject=False, read_fails=False, lose_applied_read=False):
        code = launch.locomotion_dock_script(world_start_xy=[0.,0.], world_start_yaw_rad=0., dock_world_xy=[1.,0.], dock_world_yaw_rad=0., dock_position_only=False, refine_world_xy=None, refine_world_yaw_rad=None, refine_position_only=False, dock_position_tolerance_m=.1, refine_position_tolerance_m=.1, timeout_s=10.)
        tree = ast.parse(code)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Dock")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "stand")
        calls = []
        state = {"mode": initial}
        class Clock:
            def now(self): return NS(to_msg=lambda: NS(sec=1, nanosec=0))
        def call(node, client, request, **kwargs):
            calls.append(client)
            if client == "get":
                if read_fails: raise RuntimeError("service timeout")
                if lose_applied_read and calls.count("set") and not state.get("lost_applied_read"):
                    state["lost_applied_read"] = True
                    raise namespace["ModeHandshakeError"]("service timeout: /GetMcAction")
                return NS(header=NS(code=0), info=NS(action_desc=state["mode"], status=NS(value=100)))
            if not reject: state["mode"] = request.command.action_desc
            if lose_ack: raise RuntimeError("service timeout")
            return NS(response=NS(status=NS(value=0 if reject else 1), header=NS(code=0), message="rejected" if reject else "ok"))
        namespace = dict(time=__import__("time"), SetMcAction=NS(Request=NS), GetMcAction=NS(Request=lambda: NS(request=NS(header=NS()))), RequestHeader=NS, McActionCommand=NS, SOURCE="test", call=call, message_to_ordereddict=lambda m: {})
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name in ("time", "json") for a in n.names):
                exec(compile(ast.Module(body=[n], type_ignores=[]), "generated", "exec"), namespace)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("infra_elapsed", "ensure_mode", "ModeHandshakeError"):
                exec(compile(ast.Module(body=[n], type_ignores=[]), "generated", "exec"), namespace)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "generated", "exec"), namespace)
        node = NS(mode="set", get_mode="get", mode_requests=[], get_clock=lambda: Clock())
        error = None
        try: namespace["stand"](node)
        except RuntimeError as exc: error = exc
        return calls, node, error

    def test_already_standing_does_not_repeat_set(self):
        calls, node, error = self.invoke()
        self.assertIsNone(error)
        self.assertEqual(calls.count("set"), 0)
        self.assertGreaterEqual(calls.count("get"), 1)

    def test_lost_reply_requires_applied_readback_without_resending(self):
        calls, node, error = self.invoke("PASSIVE_DEFAULT", lose_ack=True)
        self.assertIsNone(error)
        self.assertEqual(calls.count("set"), 1)
        self.assertGreaterEqual(calls.count("get"), 2)

    def test_rejection_is_not_hidden_by_readback(self):
        calls, node, error = self.invoke("PASSIVE_DEFAULT", reject=True)
        self.assertIsNotNone(error)
        self.assertIn("MODE_REQUEST_REJECTED", str(error))
        self.assertEqual(calls.count("set"), 1)

    def test_failed_readback_never_sends_set(self):
        calls, node, error = self.invoke(read_fails=True)
        self.assertIsNotNone(error)
        self.assertEqual(calls.count("set"), 0)

    def test_transient_applied_read_timeout_never_resends_set(self):
        calls, node, error = self.invoke("PASSIVE_DEFAULT", lose_applied_read=True)
        self.assertIsNone(error)
        self.assertEqual(calls.count("set"), 1)
        self.assertEqual(calls.count("get"), 3)
        self.assertEqual(len(node.mode_requests[0]["verification"]["read_transport_errors"]), 1)


    def test_transition_is_not_an_applied_mode(self):
        from scripts.research.issue46.native_runtime.mode_handshake import ensure_mode
        rows = iter([dict(name="STAND_DEFAULT", status=200, header_code=0),
                     dict(name="STAND_DEFAULT", status=100, header_code=0)])
        sent=[]
        result=ensure_mode("STAND_DEFAULT", lambda _:next(rows), lambda _:sent.append(True))
        self.assertEqual(sent, [])
        self.assertEqual(result["applied"]["status"],100)
        self.assertEqual(result["outcome"],"APPLIED_NO_SET")

    def test_other_mode_transition_cannot_be_interrupted(self):
        from scripts.research.issue46.native_runtime.mode_handshake import ensure_mode, ModeHandshakeError
        sent=[]
        with self.assertRaisesRegex(ModeHandshakeError,"MODE_TRANSITION_NOT_READY"):
            ensure_mode("STAND_DEFAULT", lambda _:dict(name="PASSIVE_DEFAULT",status=200,header_code=0), lambda _:sent.append(True))
        self.assertEqual(sent,[])

if __name__ == "__main__": unittest.main()
