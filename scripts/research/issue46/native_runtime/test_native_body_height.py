"""No-physics checks of the installed-MC body-height compatibility probe."""
import ast,builtins,dis,inspect,types,unittest
from scripts.research.issue46.native_runtime import native_body_height as body
class BodyHeightTests(unittest.TestCase):
    def test_probe_closures_have_no_missing_runtime_globals(self):
        pending=[body.run_height_probe.__code__];names=set()
        while pending:
            code=pending.pop();names.update(i.argval for i in dis.get_instructions(code) if i.opname=="LOAD_GLOBAL")
            pending.extend(x for x in code.co_consts if isinstance(x,types.CodeType))
        self.assertFalse(names-set(dir(builtins))-{"MICRO","SOURCE","write_durable_milestone"})
    def test_generated_observer_cannot_construct_velocity_publisher(self):
        code=body.build_script({},[])
        tree=ast.parse(code);dock=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Dock")
        init=next(n for n in dock.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="create_publisher" for n in ast.walk(init)))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_probe")
        cls=next(n for n in ast.walk(run) if isinstance(n,ast.ClassDef) and n.name=="Micro")
        pub=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="publish")
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="publish" for n in ast.walk(pub)))
        self.assertIn("BODY_OBSERVER_VELOCITY_FORBIDDEN",ast.unparse(pub))
    def test_probe_never_requests_action_switch_or_joint_commands(self):
        code=inspect.getsource(body.run_height_probe)
        self.assertNotIn("stand_once",code);self.assertNotIn("SetMcAction",code)
        self.assertNotIn("JointCommand",code);self.assertNotIn("mj_step",code)
        with self.assertRaises(ValueError):body.build_script({},[],.20)
