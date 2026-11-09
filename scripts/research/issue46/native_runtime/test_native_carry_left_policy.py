import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from scripts.research.issue46.native_runtime import native_carry_left_policy as policy

NAMES=tuple("left_"+str(i) for i in range(7))+tuple("right_"+str(i) for i in range(7))
LIMITS={n:(-2.,2.,3.) for n in NAMES}
FIELDS=dict(position=.2,velocity=0.,effort=0.,stiffness=40.,damping=2.)

class NativeLeftTests(unittest.TestCase):
    def packet(self):
        return dict(stamp_ns=1000000000,received_ns=2000000000,joints={n:dict(FIELDS) for n in NAMES})
    def test_source_active_and_inactive_are_distinguished(self):
        p=self.packet()
        self.assertTrue(policy.validate_native_arm_packet(p,LIMITS,2000000000,1000000000))
        for j in p["joints"].values():j.update(stiffness=0.,damping=0.)
        self.assertFalse(policy.validate_native_arm_packet(p,LIMITS,2000000000,1000000000))
    def test_stale_missing_nonfinite_and_limit_fail_closed(self):
        for mutation in ("stale","missing","nan","limit","mixed"):
            p=self.packet()
            if mutation=="stale":p["received_ns"]-=200000001
            elif mutation=="missing":p["joints"].pop(NAMES[0])
            elif mutation=="nan":p["joints"][NAMES[0]]["position"]=float("nan")
            elif mutation=="limit":p["joints"][NAMES[0]]["position"]=3.
            else:p["joints"][NAMES[0]].update(stiffness=0.,damping=0.)
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):
                policy.validate_native_arm_packet(p,LIMITS,2000000000,1000000000)
    def test_duplicate_source_does_not_refresh_receipt(self):
        n=SimpleNamespace(native_policy_stamp=1000000000,native_policy_packet=self.packet())
        m=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nanosec=0)),joints=[])
        before=copy.deepcopy(n.native_policy_packet)
        policy.on_native_carry_arm(n,m)
        self.assertEqual(n.native_policy_packet,before)
    def test_enable_is_scoped_to_carry_after_qualified_turn(self):
        n=SimpleNamespace(lift_phase={})
        self.assertIsNone(policy.select_native_carry_left(n,"APPROACH"))
        n.lift_phase["placement_active"]=True
        self.assertIsNone(policy.select_native_carry_left(n,"CARRY_HOLD"))
    def test_generated_extension_does_not_create_a_public_publisher(self):
        from pathlib import Path
        source=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261109T051058Z/native_hal_bilateral_arm_pregrasp.py").read_text()
        code=policy.extend_unloaded_arm_follow(source)
        compile(code,"native_left_owner","exec")
        self.assertEqual(code.count("self.pub=self.create_publisher("),1)
        self.assertIn("native_left=select_native_carry_left(self,stage)",code)
        self.assertIn("if native_left and name in LEFT:",code)
        self.assertEqual(code.count("def select_native_carry_left"),1)

    def test_only_left_follows_bounded_verified_native_commands(self):
        import json,os
        from pathlib import Path
        left,right=NAMES[:7],NAMES[7:]
        def endpoint(name):
            return SimpleNamespace(node_name=name,node_namespace="/",topic_type="aimdk_msgs/msg/JointCommandArray")
        n=SimpleNamespace(lift_phase={},native_generation=1,native_policy_packet=self.packet(),
            arm={x:0. for x in NAMES},native_row={"joints":{x:{"bias_force":.3} for x in NAMES}},
            last_trace_at=2.,get_name=lambda:"owner",
            get_publishers_info_by_topic=lambda topic:[endpoint("mc_ros2_node123" if "/mc/" in topic else "owner")])
        marker=dict(owner_pid=os.getpid(),generation=1,turn_and_composite_stop_pass=True,created_ns=1900000000,deadline_ns=3000000000)
        def read(path,*args,**kwargs):
            return json.dumps(marker if path.name=="carry_unloaded_arm_follow.json" else {"expected_mc_node":"mc_ros2_node123"})
        with patch.object(Path,"exists",return_value=True),patch.object(Path,"read_text",read),patch.object(policy.time,"monotonic_ns",return_value=2000000000),patch.object(policy.time,"time_ns",return_value=1000000000),patch.object(policy.time,"monotonic",return_value=2.),patch.multiple(policy,LEFT=left,RIGHT=right,LIFT_CONTRACT={"retract_preflight":{"source_motion_limits":LIMITS}},GAINS={x:(80.,4.) for x in NAMES},create=True):
            chosen=policy.select_native_carry_left(n,"CARRY_HOLD")
            self.assertEqual(set(chosen),set(left))
            self.assertTrue(all(abs(j["position"]-.01)<1e-10 for j in chosen.values()))
            self.assertTrue(all(j["stiffness"]==40. for j in chosen.values()))
            n.get_publishers_info_by_topic=lambda topic:[endpoint("arbitrary_node")]
            with self.assertRaisesRegex(RuntimeError,"SOURCE_IDENTITY"):
                policy.select_native_carry_left(n,"CARRY_HOLD")
