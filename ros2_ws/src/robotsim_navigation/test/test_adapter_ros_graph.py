"""Isolated ROS graph test for the proposal-only point-goal adapter.

Run only in a network-isolated container with a private ROS_DOMAIN_ID and
ROBOTSIM_ISOLATED_ROS_GRAPH=1. This guard prevents accidental interaction with
the live MC/MuJoCo graph when somebody runs colcon test on a workstation.
"""
from __future__ import annotations

import os
import math
import subprocess
import threading
import time
import unittest

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Empty, String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class IsolatedRosGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ROS_AVAILABLE:
            raise unittest.SkipTest("ROS 2 Humble message/runtime packages are unavailable")
        if os.environ.get("ROBOTSIM_ISOLATED_ROS_GRAPH") != "1":
            raise unittest.SkipTest("requires explicit network-isolated ROS graph opt-in")
        if not os.environ.get("ROS_DOMAIN_ID") or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
            raise RuntimeError("isolated test requires ROS_DOMAIN_ID and ROS_LOCALHOST_ONLY=1")

        rclpy.init(args=None)
        cls.node = rclpy.create_node("issue80_adapter_graph_probe")
        cls.lock = threading.Lock()
        cls.twists = []
        cls.statuses = []
        cls.sequence = 0
        cls.last_stamp = (1_792_500_000, 0)
        cls.height_override = 0.65
        cls.proposal_sub = cls.node.create_subscription(
            Twist, "/robotsim/navigation/cmd_vel_proposal", cls._on_twist, 10
        )
        cls.status_sub = cls.node.create_subscription(
            String, "/robotsim/navigation/status", cls._on_status, 10
        )
        cls.odom_pub = cls.node.create_publisher(Odometry, "/aima/hal/odom/state", 10)
        cls.goal_pub = cls.node.create_publisher(PoseStamped, "/robotsim/navigation/goal", 10)
        cls.cancel_pub = cls.node.create_publisher(Empty, "/robotsim/navigation/cancel", 10)

        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.node)
        cls.executor_thread = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.executor_thread.start()

        cls.node_log = open("/tmp/robotsim_issue80_adapter_node.log", "w", encoding="utf-8")
        cls.adapter = subprocess.Popen(
            [
                "ros2", "run", "robotsim_navigation", "point_goal_adapter",
                "--ros-args",
                "-p", "pose_timeout_s:=0.20",
                "-p", "goal_timeout_s:=0.90",
                "-p", "arrival_dwell_s:=0.05",
                "-p", "stop_hold_s:=0.15",
            ],
            env=os.environ.copy(),
            stdout=cls.node_log,
            stderr=subprocess.STDOUT,
        )
        cls._wait_for_graph()
        cls.heartbeat_enabled = threading.Event()
        cls.heartbeat_enabled.set()
        cls.heartbeat_stop = threading.Event()
        cls.heartbeat_thread = threading.Thread(target=cls._heartbeat, daemon=True)
        cls.heartbeat_thread.start()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "heartbeat_stop", None):
            cls.heartbeat_stop.set()
            cls.heartbeat_thread.join(timeout=1.0)
        if getattr(cls, "adapter", None) and cls.adapter.poll() is None:
            cls.adapter.terminate()
            try:
                cls.adapter.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                cls.adapter.kill()
                cls.adapter.wait(timeout=2.0)
        if getattr(cls, "node_log", None):
            cls.node_log.close()
        if getattr(cls, "executor", None):
            cls.executor.shutdown(timeout_sec=2.0)
        if getattr(cls, "node", None):
            cls.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @classmethod
    def _on_twist(cls, message):
        with cls.lock:
            cls.twists.append((time.monotonic(), message.linear.x, message.angular.z))

    @classmethod
    def _on_status(cls, message):
        with cls.lock:
            cls.statuses.append((time.monotonic(), message.data))

    @classmethod
    def _wait_for_graph(cls):
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if cls.adapter.poll() is not None:
                raise AssertionError("adapter process exited during startup:\n" + cls._read_log())
            names = set(cls.node.get_node_names())
            if "robotsim_point_goal_adapter" in names:
                if (
                    cls.odom_pub.get_subscription_count() > 0
                    and cls.goal_pub.get_subscription_count() > 0
                    and cls.cancel_pub.get_subscription_count() > 0
                    and cls.node.count_publishers(
                        "/robotsim/navigation/cmd_vel_proposal"
                    ) > 0
                    and cls.node.count_publishers(
                        "/robotsim/navigation/status"
                    ) > 0
                ):
                    return
            time.sleep(0.05)
        raise AssertionError("adapter endpoints did not appear:\n" + cls._read_log())

    @classmethod
    def _read_log(cls):
        cls.node_log.flush()
        with open("/tmp/robotsim_issue80_adapter_node.log", encoding="utf-8") as stream:
            return stream.read()

    @classmethod
    def _next_stamp(cls):
        with cls.lock:
            sec, nanosec = cls.last_stamp
            nanosec += 20_000_000
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            cls.last_stamp = (sec, nanosec)
            return cls.last_stamp

    @classmethod
    def _publish_odom(
        cls, *, frame="odom", child="base_link", z=None, roll_rad=0.0, stamp=None
    ):
        message = Odometry()
        message.header.frame_id = frame
        message.child_frame_id = child
        sec, nanosec = stamp if stamp is not None else cls._next_stamp()
        message.header.stamp.sec = sec
        message.header.stamp.nanosec = nanosec
        message.pose.pose.position.z = cls.height_override if z is None else z
        message.pose.pose.orientation.x = math.sin(roll_rad / 2.0)
        message.pose.pose.orientation.w = math.cos(roll_rad / 2.0)
        cls.odom_pub.publish(message)
        return (sec, nanosec)

    @classmethod
    def _heartbeat(cls):
        while not cls.heartbeat_stop.is_set():
            if cls.heartbeat_enabled.is_set():
                cls._publish_odom()
            time.sleep(0.035)

    def _counts(self):
        with self.lock:
            return len(self.twists), len(self.statuses)

    def _wait_status(self, fragment, after, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.adapter.poll() is not None:
                self.fail("adapter exited:\n" + self._read_log())
            with self.lock:
                found = [item for item in self.statuses[after:] if fragment in item[1]]
            if found:
                return found[-1]
            time.sleep(0.01)
        with self.lock:
            recent = self.statuses[-12:]
        self.fail(f"did not receive status containing {fragment!r}; recent={recent!r}")

    def _wait_twist(self, predicate, after, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                found = [item for item in self.twists[after:] if predicate(item[1], item[2])]
            if found:
                return found[-1]
            time.sleep(0.01)
        with self.lock:
            recent = self.twists[-12:]
        self.fail(f"no matching Twist after event index {after}; recent={recent!r}")

    def _send_goal(self, *, frame="odom", x=2.0):
        goal = PoseStamped()
        goal.header.frame_id = frame
        goal.header.stamp.sec = 1_792_500_000
        goal.pose.position.x = x
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

    def _begin_goal_and_wait_for_motion(self):
        time.sleep(0.18)  # let the adapter's post-stop goal hold expire
        twist_index, status_index = self._counts()
        self._send_goal()
        self._wait_status("GOAL_ACCEPTED", status_index)
        self._wait_twist(lambda linear, _angular: linear > 0.0, twist_index)
        return twist_index

    def test_adapter_ros_graph_contract_and_stop_gates(self):
        # Confirm the adapter exposes only the proposal output, not MC velocity topics/services.
        name_pairs = self.node.get_node_names_and_namespaces()
        adapter_name, adapter_namespace = next(
            pair for pair in name_pairs if pair[0] == "robotsim_point_goal_adapter"
        )
        publishers = self.node.get_publisher_names_and_types_by_node(
            adapter_name, adapter_namespace
        )
        publisher_topics = {name for name, _types in publishers}
        self.assertIn("/robotsim/navigation/cmd_vel_proposal", publisher_topics)
        self.assertNotIn("/cmd_vel", publisher_topics)
        self.assertNotIn("/aima/mc/locomotion/velocity", publisher_topics)

        # Explicit ROS remaps to either common live command path must fail before
        # the adapter creates a publisher on the remapped name.
        for live_topic in ("/cmd_vel", "/aima/mc/locomotion/velocity"):
            with self.subTest(remap_target=live_topic):
                probe = subprocess.run(
                    [
                        "ros2", "run", "robotsim_navigation", "point_goal_adapter",
                        "--ros-args", "-r",
                        f"/robotsim/navigation/cmd_vel_proposal:={live_topic}",
                    ],
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=False,
                )
                self.assertNotEqual(probe.returncode, 0, probe.stdout + probe.stderr)
                self.assertIn(
                    "adapter must remain disconnected from live MC topics",
                    probe.stdout + probe.stderr,
                )

        services = self.node.get_service_names_and_types_by_node(
            adapter_name, adapter_namespace
        )
        service_names = {name for name, _types in services}
        self.assertFalse(any("/aima/mc" in name or "locomotion" in name for name in service_names))

        # Real Odometry + PoseStamped produce a bounded proposal on the isolated topic.
        motion_index = self._begin_goal_and_wait_for_motion()
        with self.lock:
            self.assertTrue(any(
                linear <= 0.20 and abs(angular) <= 0.30
                for _time, linear, angular in self.twists[motion_index:]
            ))

        # A wrong-frame goal is rejected without any nonzero proposal.
        cancel_index = self._counts()[1]
        self.cancel_pub.publish(Empty())
        self._wait_status("CANCELLED", cancel_index)
        time.sleep(0.18)
        twist_index, status_index = self._counts()
        self._send_goal(frame="map")
        self._wait_status("GOAL_REJECTED:FRAME_MISMATCH", status_index)
        time.sleep(0.12)
        with self.lock:
            self.assertFalse(any(
                abs(linear) > 1e-9 or abs(angular) > 1e-9
                for _time, linear, angular in self.twists[twist_index:]
            ))

        # A wrong Odometry frame during an active goal immediately stops the proposal.
        self._begin_goal_and_wait_for_motion()
        self.heartbeat_enabled.clear()
        time.sleep(0.08)
        twist_index, status_index = self._counts()
        self._publish_odom(frame="map")
        self._wait_status("ODOM_REJECTED:FRAME_MISMATCH", status_index)
        self._wait_twist(lambda linear, angular: abs(linear) < 1e-9 and abs(angular) < 1e-9,
                         twist_index)
        self.heartbeat_enabled.set()

        # A repeated Odometry timestamp is rejected and stops the active goal.
        self._begin_goal_and_wait_for_motion()
        self.heartbeat_enabled.clear()
        time.sleep(0.08)
        with self.lock:
            duplicate_stamp = self.last_stamp
        twist_index, status_index = self._counts()
        self._publish_odom(stamp=duplicate_stamp)
        self._wait_status("ODOM_REJECTED:STAMP_REGRESSION", status_index)
        self._wait_twist(lambda linear, angular: abs(linear) < 1e-9 and abs(angular) < 1e-9,
                         twist_index)
        self.heartbeat_enabled.set()

        # Unsafe height is rejected while a goal is active and emits zero Twist.
        self._begin_goal_and_wait_for_motion()
        self.heartbeat_enabled.clear()
        time.sleep(0.08)
        twist_index, status_index = self._counts()
        self._publish_odom(z=0.20)
        self._wait_status("ODOM_REJECTED:UNSAFE_HEIGHT", status_index)
        self._wait_twist(lambda linear, angular: abs(linear) < 1e-9 and abs(angular) < 1e-9,
                         twist_index)
        self.heartbeat_enabled.set()

        self._begin_goal_and_wait_for_motion()
        self.heartbeat_enabled.clear()
        time.sleep(0.08)
        twist_index, status_index = self._counts()
        self._publish_odom(roll_rad=0.5)
        self._wait_status("ODOM_REJECTED:UNSAFE_TILT", status_index)
        self._wait_twist(lambda linear, angular: abs(linear) < 1e-9 and abs(angular) < 1e-9,
                         twist_index)
        self.heartbeat_enabled.set()

        # Cancellation and stale received Odometry each stop active proposals.
        self._begin_goal_and_wait_for_motion()
        twist_index, status_index = self._counts()
        self.cancel_pub.publish(Empty())
        self._wait_status("CANCELLED", status_index)
        self._wait_twist(lambda linear, angular: abs(linear) < 1e-9 and abs(angular) < 1e-9,
                         twist_index)
        time.sleep(0.25)
        with self.lock:
            idle_after_cancel = self.twists[twist_index:]
        self.assertGreaterEqual(len(idle_after_cancel), 4)
        self.assertTrue(all(
            abs(linear) < 1e-9 and abs(angular) < 1e-9
            for _time, linear, angular in idle_after_cancel
        ))

        self._begin_goal_and_wait_for_motion()
        self.heartbeat_enabled.clear()
        twist_index, status_index = self._counts()
        self._wait_status("STOPPED:STALE_ODOMETRY", status_index, timeout=1.0)
        self._wait_twist(lambda linear, angular: abs(linear) < 1e-9 and abs(angular) < 1e-9,
                         twist_index)
        self.heartbeat_enabled.set()

        # Goal timeout stops output even while valid Odometry continues.
        self._begin_goal_and_wait_for_motion()
        twist_index, status_index = self._counts()
        self._wait_status("STOPPED:TIMED_OUT", status_index, timeout=1.5)
        self._wait_twist(lambda linear, angular: abs(linear) < 1e-9 and abs(angular) < 1e-9,
                         twist_index)


if __name__ == "__main__":
    unittest.main()
