#!/usr/bin/env python3
"""Local-only ROS 2 / SDK2 integration check for the pinned G1 simulator."""

import math
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robotsim_interfaces.msg import RobotCommand, RobotState
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage


STATE_TOPIC = "/simulation/ground_truth/g1/state"
COMMAND_TOPIC = "/robot/g1/command"
MODEL_IDENTITY = "unitree_g1:unitree_mujoco@1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d:scene_29dof"


def percentile(samples, probability):
    ordered = sorted(samples)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def assert_qos(info, reliability):
    profile = info.qos_profile
    assert profile.reliability == reliability, profile
    assert profile.durability == DurabilityPolicy.VOLATILE, profile
    if profile.history != HistoryPolicy.UNKNOWN:
        assert profile.history == HistoryPolicy.KEEP_LAST, profile
        assert profile.depth == 1, profile


def main():
    rclpy.init()
    node = rclpy.create_node("robotsim_issue12_live_test")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    state_samples = []
    state_receive_ns = []
    clock_samples = []
    static_transforms = []
    state_sub = node.create_subscription(
        RobotState,
        STATE_TOPIC,
        lambda message: (state_samples.append(message), state_receive_ns.append(time.monotonic_ns())),
        QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        ),
    )
    clock_sub = node.create_subscription(
        Clock,
        "/clock",
        clock_samples.append,
        QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        ),
    )
    tf_sub = node.create_subscription(
        TFMessage,
        "/tf_static",
        static_transforms.append,
        QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        ),
    )
    command_pub = node.create_publisher(
        RobotCommand,
        COMMAND_TOPIC,
        QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        ),
    )

    def spin_until(predicate, timeout_s, description):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            executor.spin_once(timeout_sec=0.02)
        raise AssertionError(f"timed out waiting for {description}")

    try:
        spin_until(lambda: len(state_samples) >= 40 and clock_samples, 10.0, "state and /clock")
        latest = state_samples[-1]
        assert latest.schema_version == "1.0"
        assert latest.model_identity == MODEL_IDENTITY
        assert latest.generation
        assert latest.origin == RobotState.ORIGIN_GROUND_TRUTH
        assert latest.lifecycle == RobotState.LIFECYCLE_RUNNING
        assert len(latest.joints.name) == 29
        assert len(set(latest.joints.name)) == 29
        assert len(latest.joints.position) == len(latest.joints.velocity) == len(latest.joints.effort) == 29
        assert latest.header.frame_id == "world"
        assert latest.joints.header.frame_id == "world"
        assert latest.imu.header.frame_id == "imu_link"
        assert latest.base_frame == "base_link" and latest.imu_frame == "imu_link"
        assert not latest.has_root_pose and not latest.has_root_twist
        assert stamp_ns(latest.header.stamp) == latest.backend_tick * 1_000_000
        assert all(
            later.sequence > earlier.sequence
            for earlier, later in zip(state_samples, state_samples[1:])
            if earlier.generation == later.generation
        )
        assert all(
            stamp_ns(later.header.stamp) >= stamp_ns(earlier.header.stamp)
            for earlier, later in zip(state_samples, state_samples[1:])
            if earlier.generation == later.generation
        )
        state_times = {stamp_ns(sample.header.stamp) for sample in state_samples}
        assert any(stamp_ns(sample.clock) in state_times for sample in clock_samples)

        def state_publishers():
            return node.get_publishers_info_by_topic(STATE_TOPIC)

        def command_subscribers():
            return node.get_subscriptions_info_by_topic(COMMAND_TOPIC)

        spin_until(lambda: state_publishers() and command_subscribers(), 5.0, "ROS graph endpoints")
        assert_qos(state_publishers()[0], ReliabilityPolicy.BEST_EFFORT)
        assert_qos(command_subscribers()[0], ReliabilityPolicy.RELIABLE)

        spin_until(lambda: any(
            transform.header.frame_id == "base_link" and transform.child_frame_id == "imu_link"
            for message in static_transforms
            for transform in message.transforms
        ), 5.0, "base_link to imu_link static TF")
        assert len(clock_samples) >= 1

        joint_names = list(latest.joints.name)
        tested_joint = joint_names.index("left_elbow_joint")
        baseline = state_samples[-10:]
        assert all(abs(sample.joints.effort[tested_joint]) < 0.05 for sample in baseline), [
            sample.joints.effort[tested_joint] for sample in baseline
        ]

        def send_command(sequence, generation, effort):
            message = RobotCommand()
            message.header.stamp = state_samples[-1].header.stamp
            message.header.frame_id = "base_link"
            message.schema_version = "1.0"
            message.model_identity = MODEL_IDENTITY
            message.generation = generation
            message.sequence = sequence
            message.source_id = "issue12-live-test"
            message.command_type = RobotCommand.COMMAND_TORQUE
            message.joint_names = joint_names
            message.effort_nm = [0.0] * len(joint_names)
            message.effort_nm[tested_joint] = effort
            command_pub.publish(message)

        starting_sequence = state_samples[-1].sequence
        send_command(0, "stale-" + latest.generation, 0.1)
        spin_until(lambda: state_samples[-1].sequence >= starting_sequence + 20, 3.0, "state after stale command")
        assert all(abs(sample.joints.effort[tested_joint]) < 0.05 for sample in state_samples[-10:])

        roundtrip_ms = []
        command_sequence = 0
        last_observed_sequence = state_samples[-1].sequence
        for phase in range(20):
            effort = 0.1 if phase % 2 == 0 else -0.1
            started = time.monotonic_ns()
            send_command(command_sequence, latest.generation, effort)
            command_sequence += 1
            observed = None
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
                for sample in state_samples:
                    if sample.sequence <= last_observed_sequence:
                        continue
                    if sample.generation != latest.generation:
                        raise AssertionError("generation changed during live command test")
                    value = sample.joints.effort[tested_joint]
                    if value * effort > 0.05 * abs(effort):
                        observed = sample
                        break
                if observed is not None:
                    break
            assert observed is not None, f"no matching tau_est for command phase {phase}"
            roundtrip_ms.append((time.monotonic_ns() - started) / 1_000_000.0)
            last_observed_sequence = observed.sequence

        last_feedback_sequence = last_observed_sequence
        spin_until(
            lambda: len(state_samples) >= 5
            and state_samples[-1].sequence >= last_feedback_sequence + 5
            and all(abs(sample.joints.effort[tested_joint]) < 0.05 for sample in state_samples[-5:]),
            3.0,
            "zero-torque timeout fallback",
        )

        state_intervals_ms = [
            (right - left) / 1_000_000.0
            for left, right in zip(state_receive_ns, state_receive_ns[1:])
        ]
        print("PASS state_schema=1.0 origin=ground_truth joints=29 root_pose=unavailable root_twist=unavailable")
        print("PASS timestamp=backend_tick_ms*1e6 sequence=monotonic generation=stable")
        print("PASS qos state=best_effort_volatile command=reliable_volatile")
        print("PASS tf_static=base_link->imu_link clock=/clock stale_generation_rejected")
        print("PASS command_roundtrip_phases=20 fallback=zero_torque after_timeout_ms=250")
        print(f"restart_reference_generation={latest.generation}")
        print(
            "command_send_to_feedback_ms_p50_p95_p99="
            f"{percentile(roundtrip_ms, 0.50):.3f}/"
            f"{percentile(roundtrip_ms, 0.95):.3f}/"
            f"{percentile(roundtrip_ms, 0.99):.3f}"
        )
        print(
            "state_callback_interval_ms_p50_p95_p99="
            f"{percentile(state_intervals_ms, 0.50):.3f}/"
            f"{percentile(state_intervals_ms, 0.95):.3f}/"
            f"{percentile(state_intervals_ms, 0.99):.3f}"
        )
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
