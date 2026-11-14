"""Execute the generated prearm call site with bounded transport failures."""
import ast
import contextlib
import io
import json
from types import SimpleNamespace as NS
import unittest

from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as launch


class PrearmReadinessTests(unittest.TestCase):
    def execute(self, failures=1, *, invalid=False, executor_error=False):
        tree = ast.parse(launch.prearm_script(single_request_wait_s=2.0))
        state = dict(now=0., mode="PASSIVE_DEFAULT", get=0, set=0, destroyed=False)
        clock = NS(sleep=lambda seconds: state.update(now=state["now"]+seconds))
        node = NS(create_client=lambda kind, path: "get" if "Get" in path else "set",
                  get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: NS())),
                  destroy_node=lambda: state.update(destroyed=True))
        namespace = dict(json=json, rclpy=NS(init=lambda: None, shutdown=lambda: None),
                         Node=lambda name: node, GetMcAction=NS(Request=lambda: NS()),
                         SetMcAction=NS(Request=lambda: NS()), CommonRequest=lambda: NS(header=NS()),
                         RequestHeader=lambda: NS(), McActionCommand=lambda: NS())
        # Use actual generated handshake functions, replacing only elapsed time,
        # sleep and the ROS transport. Execute the actual top-level try/finally.
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
                exec(compile(ast.Module(body=[item], type_ignores=[]), "prearm", "exec"), namespace)
        namespace.update(infra_elapsed=lambda: state["now"], _infra_time=clock)
        def call_service(node, client, request, *, timeout_s):
            if client == "get":
                state["get"] += 1
                if executor_error:
                    raise namespace["ModeHandshakeError"]("FEEDBACK_EXECUTOR_FAILURE:dead")
                if state["get"] <= failures:
                    state["now"] += timeout_s
                    raise namespace["ModeHandshakeError"]("service timeout: /GetMcAction")
                return NS(header=NS(code=1 if invalid else 0),
                          info=NS(action_desc=state["mode"], status=NS(value=100)))
            state["set"] += 1
            state["mode"] = request.command.action_desc
            return NS(response=NS(header=NS(code=0), status=NS(value=1), message="ok"))
        namespace["call_service"] = call_service
        body = [item for item in tree.body if isinstance(item, (ast.Expr, ast.Assign, ast.Try))]
        output=io.StringIO(); error=None
        with contextlib.redirect_stdout(output):
            try: exec(compile(ast.Module(body=body, type_ignores=[]), "prearm", "exec"), namespace)
            except RuntimeError as exc: error=exc
        return state, output.getvalue(), error

    def test_first_read_timeout_can_recover_before_single_set(self):
        state, output, error = self.execute()
        self.assertIsNone(error)
        self.assertEqual(state["set"], 1)
        result=json.loads(output)
        self.assertIn("transport_error", result["initialization_attempts"][0])
        self.assertEqual(result["mode_status"], 100)
        self.assertTrue(state["destroyed"])

    def test_persistent_timeout_stops_within_deadline_without_set(self):
        state, _, error=self.execute(failures=100)
        self.assertIn("MC_NOT_INITIALIZED_NO_SET", str(error))
        self.assertLessEqual(state["now"], 4.)
        self.assertEqual(state["set"], 0)
        self.assertTrue(state["destroyed"])

    def test_invalid_response_is_not_retried_or_accepted(self):
        state, _, error=self.execute(failures=0, invalid=True)
        self.assertIn("MODE_READBACK_INVALID", str(error))
        self.assertEqual(state["get"], 1)
        self.assertEqual(state["set"], 0)

    def test_executor_failure_is_not_hidden_as_startup_delay(self):
        state, _, error=self.execute(executor_error=True)
        self.assertIn("FEEDBACK_EXECUTOR_FAILURE", str(error))
        self.assertEqual(state["get"], 1)
        self.assertEqual(state["set"], 0)


if __name__ == "__main__":
    unittest.main()
