"""ROS 2 wrapper for measured-Odometry point-goal control.

The only command output is an isolated proposal topic. This node never calls MC
services and refuses the standard MC bridge topics. Do not remap it to live MC
until Issue #80 M1 has passed and that integration is separately authorized.
"""
from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from .adapter_core import (
    AdapterInputError,
    DEFAULT_BASE_FRAME,
    DEFAULT_ODOM_FRAME,
    DEFAULT_PROPOSAL_TOPIC,
    odometry_to_pose,
    pose_stamped_to_goal,
    validate_proposal_topic,
)
from .point_goal_controller import (
    ControllerConfig,
    NavState,
    PointGoalController,
)


_TERMINAL_STATES = {
    NavState.ARRIVED,
    NavState.CANCELLED,
    NavState.TIMED_OUT,
    NavState.STALE_STATE,
    NavState.FRAME_MISMATCH,
    NavState.INVALID_INPUT,
    NavState.CLOCK_DISCONTINUITY,
}


class PointGoalAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("robotsim_point_goal_adapter")
        self.declare_parameter("odom_topic", "/aima/hal/odom/state")
        self.declare_parameter("goal_topic", "/robotsim/navigation/goal")
        self.declare_parameter("cancel_topic", "/robotsim/navigation/cancel")
        self.declare_parameter("status_topic", "/robotsim/navigation/status")
        self.declare_parameter("odom_frame", DEFAULT_ODOM_FRAME)
        self.declare_parameter("base_frame", DEFAULT_BASE_FRAME)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("xy_tolerance_m", 0.12)
        self.declare_parameter("yaw_tolerance_rad", 0.12)
        self.declare_parameter("align_tolerance_rad", 0.20)
        self.declare_parameter("max_forward_mps", 0.20)
        self.declare_parameter("max_angular_rps", 0.30)
        self.declare_parameter("pose_timeout_s", 0.20)
        self.declare_parameter("goal_timeout_s", 30.0)
        self.declare_parameter("arrival_dwell_s", 0.50)
        self.declare_parameter("min_base_z_m", 0.52)
        self.declare_parameter("max_tilt_rad", math.radians(15.0))
        self.declare_parameter("stop_hold_s", 0.50)

        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.proposal_topic = DEFAULT_PROPOSAL_TOPIC
        resolved_proposal_topic = self.resolve_topic_name(self.proposal_topic)
        validate_proposal_topic(resolved_proposal_topic)
        if resolved_proposal_topic != DEFAULT_PROPOSAL_TOPIC:
            raise RuntimeError("proposal topic remapping is disabled until M1 passes")

        frequency = float(self.get_parameter("control_hz").value)
        if not math.isfinite(frequency) or frequency <= 0.0 or frequency > 50.0:
            raise ValueError("control_hz must be in (0, 50]")

        config = ControllerConfig(
            xy_tolerance_m=float(self.get_parameter("xy_tolerance_m").value),
            yaw_tolerance_rad=float(self.get_parameter("yaw_tolerance_rad").value),
            align_tolerance_rad=float(self.get_parameter("align_tolerance_rad").value),
            max_forward_mps=float(self.get_parameter("max_forward_mps").value),
            max_angular_rps=float(self.get_parameter("max_angular_rps").value),
            pose_timeout_s=float(self.get_parameter("pose_timeout_s").value),
            goal_timeout_s=float(self.get_parameter("goal_timeout_s").value),
            arrival_dwell_s=float(self.get_parameter("arrival_dwell_s").value),
        )
        self.controller = PointGoalController(config)
        self.min_base_z_m = float(self.get_parameter("min_base_z_m").value)
        self.max_tilt_rad = float(self.get_parameter("max_tilt_rad").value)
        if not math.isfinite(self.min_base_z_m) or self.min_base_z_m < 0.0:
            raise ValueError("min_base_z_m must be finite and nonnegative")
        if (
            not math.isfinite(self.max_tilt_rad)
            or self.max_tilt_rad <= 0.0
            or self.max_tilt_rad >= math.pi / 2.0
        ):
            raise ValueError("max_tilt_rad must be finite and in (0, pi/2)")
        self.pose_timeout_s = config.pose_timeout_s
        self.stop_hold_s = float(self.get_parameter("stop_hold_s").value)
        if not math.isfinite(self.stop_hold_s) or self.stop_hold_s < 0.0:
            raise ValueError("stop_hold_s must be finite and nonnegative")

        # State is transient sensor data; use best effort and a one-sample queue.
        odom_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.proposal_pub = self.create_publisher(Twist, self.proposal_topic, command_qos)
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value),
            self._on_odom, odom_qos,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, str(self.get_parameter("goal_topic").value),
            self._on_goal, 10,
        )
        self.cancel_sub = self.create_subscription(
            Empty, str(self.get_parameter("cancel_topic").value),
            self._on_cancel, 10,
        )

        self._latest_pose = None
        self._last_odom_stamp_ns = None
        self._active_goal = None
        self._goal_started_steady = None
        self._stop_until_steady = 0.0
        self._last_status = None
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._timer = self.create_timer(
            1.0 / frequency, self._on_timer, clock=self._steady_clock
        )
        self.get_logger().warning(
            "OFFLINE/PROPOSAL OUTPUT ONLY: no MC service or MC velocity topic is used; "
            f"Twist proposals publish on {self.proposal_topic}"
        )

    def _status(self, value: str) -> None:
        if value == self._last_status:
            return
        self._last_status = value
        message = String()
        message.data = value
        self.status_pub.publish(message)

    def _publish_twist(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        message = Twist()
        message.linear.x = float(linear_x)
        message.angular.z = float(angular_z)
        self.proposal_pub.publish(message)

    def _on_odom(self, message: Odometry) -> None:
        now = time.monotonic()
        try:
            sample = odometry_to_pose(
                message,
                received_steady_s=now,
                expected_odom_frame=self.odom_frame,
                expected_base_frame=self.base_frame,
                last_stamp_ns=self._last_odom_stamp_ns,
                min_base_z_m=self.min_base_z_m,
                max_tilt_rad=self.max_tilt_rad,
            )
        except AdapterInputError as exc:
            self._latest_pose = None
            if self._active_goal is not None:
                self._finish(f"ODOM_REJECTED:{exc.code}")
            else:
                self._status(f"ODOM_REJECTED:{exc.code}")
            return

        self._latest_pose = sample.pose
        self._last_odom_stamp_ns = sample.stamp_ns

    def _on_goal(self, message: PoseStamped) -> None:
        if self._active_goal is not None or time.monotonic() < self._stop_until_steady:
            self._status("GOAL_REJECTED:BUSY_OR_STOP_HOLD_ACTIVE")
            return
        try:
            goal = pose_stamped_to_goal(
                message, expected_frame=self.odom_frame
            )
        except AdapterInputError as exc:
            self._status(f"GOAL_REJECTED:{exc.code}")
            return

        self.controller.reset()
        self._active_goal = goal
        self._goal_started_steady = time.monotonic()
        self._status("GOAL_ACCEPTED")

    def _on_cancel(self, _message: Empty) -> None:
        if self._active_goal is not None:
            self._finish("CANCELLED")
        else:
            self._status("CANCEL_IGNORED:NO_ACTIVE_GOAL")

    def _finish(self, status: str) -> None:
        self._active_goal = None
        self._goal_started_steady = None
        self.controller.reset()
        self._publish_twist()
        self._stop_until_steady = time.monotonic() + self.stop_hold_s
        self._status(status)

    def _on_timer(self) -> None:
        now = time.monotonic()
        if self._active_goal is not None:
            if self._latest_pose is None:
                if now - self._goal_started_steady > self.pose_timeout_s:
                    self._finish("STOPPED:STALE_OR_MISSING_ODOMETRY")
                else:
                    self._publish_twist()
                    self._status("WAITING_FOR_ODOMETRY")
                return
            if now - self._latest_pose.received_steady_s > self.pose_timeout_s:
                self._finish("STOPPED:STALE_ODOMETRY")
                return

            output = self.controller.step(
                self._latest_pose,
                self._active_goal,
                now_steady_s=now,
            )
            if output.state in _TERMINAL_STATES:
                self._finish(f"STOPPED:{output.state.value.upper()}")
                return
            self._publish_twist(output.linear_x, output.angular_z)
            self._status(output.state.value.upper())
            return

        # Publish zeros continuously while idle so no consumer is left with
        # the last nonzero proposal merely because this node went quiet.
        self._publish_twist()

    def safe_shutdown(self) -> None:
        # These are isolated proposal-topic zeros. They are not a live MC stop.
        for _ in range(10):
            self._publish_twist()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PointGoalAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.safe_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
