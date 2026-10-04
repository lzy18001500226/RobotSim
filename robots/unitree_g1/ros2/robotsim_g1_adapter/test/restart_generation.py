#!/usr/bin/env python3
"""Observe the next G1 adapter generation after a controlled process restart."""

import argparse
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robotsim_interfaces.msg import RobotState

MODEL_IDENTITY = "unitree_g1:unitree_mujoco@1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d:scene_29dof"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_generation")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("robotsim_issue12_restart_observer")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    received = []
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    subscription = node.create_subscription(
        RobotState,
        "/simulation/ground_truth/g1/state",
        received.append,
        qos,
    )
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            if received and received[-1].generation != args.old_generation:
                state = received[-1]
                assert state.model_identity == MODEL_IDENTITY
                print(
                    "PASS restart_generation_changed="
                    f"{state.generation} first_received_sequence={state.sequence}"
                )
                return
        raise AssertionError("no new adapter generation observed")
    finally:
        node.destroy_subscription(subscription)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
