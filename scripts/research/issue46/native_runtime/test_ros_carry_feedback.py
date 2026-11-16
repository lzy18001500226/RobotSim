"""Generated-message transport test, isolated ROS graph; no MC or physics."""
import threading
import time
import unittest
from unittest.mock import patch

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from aimdk_msgs.msg import JointCommand, JointCommandArray
except ImportError:
    rclpy = None

from scripts.research.issue46.native_runtime import native_grasp_stand_restore as relay


@unittest.skipIf(rclpy is None, "requires isolated ROS 2 Humble + aimdk_msgs")
class CarryFeedbackTransportTests(unittest.TestCase):
    def test_main_work_does_not_block_relay_and_source_loss_still_stops(self):
        rclpy.init()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        source = Node("isolated_waist_source")
        owner = Node("isolated_carry_owner")
        source_pub = source.create_publisher(JointCommandArray, "/isolated/waist/input", qos)
        owner.waist_pub = owner.create_publisher(JointCommandArray, "/isolated/waist/output", qos)
        owner.mc_waist_blend_started = None
        owner.mc_waist_report_at = float("inf")
        owner.native_generation = 1
        received = []
        source.create_subscription(JointCommandArray, "/isolated/waist/output",
                                   lambda msg: received.append((time.monotonic(), msg)), qos)
        enabled = [True]
        def publish():
            if not enabled[0]:
                return
            msg = JointCommandArray()
            msg.header.stamp = source.get_clock().now().to_msg()
            for name in relay.WAIST:
                joint = JointCommand()
                joint.name = name
                joint.position = .01
                joint.velocity = .02
                joint.effort = .03
                joint.stiffness = 20.
                joint.damping = 2.
                msg.joints.append(joint)
            source_pub.publish(msg)
        source.create_timer(.002, publish)
        source_executor = SingleThreadedExecutor()
        source_executor.add_node(source)
        thread = threading.Thread(target=source_executor.spin, daemon=True)
        contract = {"retract_preflight": {"source_motion_limits":
                    {name: (-1., 1., 5.) for name in relay.WAIST}}}
        with patch.object(relay, "rclpy", rclpy, create=True), \
             patch.object(relay, "USE_NATIVE_FEEDBACK", True, create=True), \
             patch.object(relay, "LIFT_CONTRACT", contract, create=True):
            owner.create_subscription(JointCommandArray, "/isolated/waist/input",
                lambda msg: relay.receive_owner_waist(owner, msg), qos)
            relay.start_owner_feedback(owner)
            thread.start()
            try:
                deadline = time.monotonic() + 5.
                while not hasattr(owner, "mc_waist_receipt") and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertTrue(hasattr(owner, "mc_waist_receipt"))
                with owner.owner_waist_lock:
                    owner.mc_waist_blend_from = {name: .01 for name in relay.WAIST}
                    owner.mc_waist_blend_started = time.monotonic() - 5.
                time.sleep(.1)
                start = time.monotonic()
                # Longer than the actual 334ms interruption; no main-loop spin.
                time.sleep(.40)
                samples = [(t, msg) for t, msg in received if t >= start]
                self.assertGreater(len(samples), 80)
                max_gap = max(b[0] - a[0] for a, b in zip(samples, samples[1:]))
                self.assertLess(max_gap, .20)
                relay.request_native_waist_handoff(owner)
                msg = samples[-1][1]
                self.assertEqual([j.name for j in msg.joints], list(relay.WAIST))
                for joint in msg.joints:
                    self.assertEqual((joint.position, joint.velocity, joint.effort,
                                      joint.stiffness, joint.damping), (.01, .02, .03, 20., 2.))
                self.assertIs(owner.executor, owner.owner_feedback_executor)
                self.assertIn(owner, owner.owner_feedback_executor.get_nodes())
                enabled[0] = False
                time.sleep(.25)
                with self.assertRaisesRegex(RuntimeError, "MC_WAIST_RELAY_STALE"):
                    relay.request_native_waist_handoff(owner)
                # A real new packet resumes forwarding; the old gap remains
                # diagnostic instead of killing the callback executor.
                previous=owner.mc_waist_receipt
                enabled[0] = True
                deadline = time.monotonic() + 1.
                while owner.mc_waist_receipt==previous and time.monotonic()<deadline:
                    time.sleep(.005)
                relay.check_owner_feedback(owner)
                self.assertGreater(owner.mc_waist_receipt,previous)
                self.assertGreater(owner.mc_waist_last_recovered_gap_s,.25)
                print({"isolated_generated_transport": "PASS",
                       "main_loop_block_s": .40, "forwarded_samples_during_block": len(samples),
                       "max_forward_gap_s": max_gap, "source_loss_hard_stop_s": .20,
                       "no_mc_or_mujoco": True}, flush=True)
            finally:
                relay.stop_owner_feedback(owner)
                source_executor.shutdown(timeout_sec=1.)
                thread.join(timeout=1.)
                self.assertFalse(thread.is_alive())
                owner.destroy_node()
                source.destroy_node()
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
