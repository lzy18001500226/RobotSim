#!/usr/bin/env python3
"""Minimal official-style X2 velocity publisher with MuJoCo-odom stop."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader
from aimdk_msgs.srv import SetMcInputSource
from nav_msgs.msg import Odometry
from rclpy.node import Node


class DirectTruthRoute(Node):
    """Reuse the vendor ADD + 50 Hz velocity contract; only stopping uses odom."""

    def __init__(self, target_distance_m: float) -> None:
        super().__init__("issue80_vendor_direct_truth_route")
        self.target_distance_m = target_distance_m
        self.publisher = self.create_publisher(
            McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10
        )
        self.client = self.create_client(
            SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource"
        )
        self.start_xy: tuple[float, float] | None = None
        self.last_xy: tuple[float, float] | None = None
        self.done = False
        self.create_subscription(Odometry, "/aima/hal/odom/state", self._odom, 10)
        self.timer = self.create_timer(0.02, self._publish)

    def _odom(self, msg: Odometry) -> None:
        xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))
        if self.start_xy is None:
            self.start_xy = xy
        self.last_xy = xy
        if self.start_xy is not None:
            if math.hypot(xy[0] - self.start_xy[0], xy[1] - self.start_xy[1]) >= self.target_distance_m:
                self.done = True

    def register(self) -> bool:
        if not self.client.wait_for_service(timeout_sec=5.0):
            return False
        request = SetMcInputSource.Request()
        request.action.value = 1001
        request.input_source.name = "node"
        request.input_source.priority = 40
        request.input_source.timeout = 1000
        request.request.header.stamp = self.get_clock().now().to_msg()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        return bool(future.done() and future.result() and future.result().response.header.code == 0)

    def _publish(self) -> None:
        message = McLocomotionVelocity()
        message.header = MessageHeader()
        message.header.stamp = self.get_clock().now().to_msg()
        message.source = "node"
        message.forward_velocity = 0.0 if self.done else 0.2
        message.lateral_velocity = 0.0
        message.angular_velocity = 0.0
        self.publisher.publish(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()
    rclpy.init()
    node = DirectTruthRoute(args.distance_m)
    try:
        if not node.register():
            return 2
        deadline = time.monotonic() + args.timeout_s
        while rclpy.ok() and time.monotonic() < deadline and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.done = True
        for _ in range(50):
            node._publish()
            rclpy.spin_once(node, timeout_sec=0.02)
        return 0 if node.last_xy is not None else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
