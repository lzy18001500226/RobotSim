"""Offline adapter tests using actual ROS field names and recorded M1 values."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from robotsim_navigation.adapter_core import (
    AdapterInputError,
    DEFAULT_PROPOSAL_TOPIC,
    odometry_to_pose,
    pose_stamped_to_goal,
    validate_proposal_topic,
)
from robotsim_navigation.point_goal_controller import (
    ControllerConfig,
    NavState,
    PointGoalController,
)


FIXTURE = Path(__file__).parent / "fixtures" / "recorded_odom_samples.json"


def make_odom(sample):
    return NS(
        header=NS(
            stamp=NS(sec=sample["header"]["sec"], nanosec=sample["header"]["nanosec"]),
            frame_id=sample["header"]["frame_id"],
        ),
        child_frame_id=sample["child_frame_id"],
        pose=NS(
            pose=NS(
                position=NS(**sample["position"]),
                orientation=NS(
                    x=sample["orientation_xyzw"][0],
                    y=sample["orientation_xyzw"][1],
                    z=sample["orientation_xyzw"][2],
                    w=sample["orientation_xyzw"][3],
                ),
            )
        ),
        twist=NS(twist=NS(linear=NS(x=0.0, y=0.0, z=0.0),
                          angular=NS(x=0.0, y=0.0, z=0.0))),
    )


def make_goal(frame="odom", *, x=0.5, y=0.0, z=0.0, q=(0.0, 0.0, 0.0, 1.0)):
    return NS(
        header=NS(stamp=NS(sec=10, nanosec=25), frame_id=frame),
        pose=NS(
            position=NS(x=x, y=y, z=z),
            orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3]),
        ),
    )


class RecordedOdometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = json.loads(FIXTURE.read_text(encoding="utf-8"))["samples"]

    def test_recorded_post_reset_upright_sample_accepted(self):
        sample = self.samples[0]
        parsed = odometry_to_pose(make_odom(sample), received_steady_s=1.0)
        self.assertEqual(parsed.pose.frame_id, "odom")
        self.assertAlmostEqual(parsed.pose.x, sample["position"]["x"])
        self.assertAlmostEqual(parsed.pose.y, sample["position"]["y"])
        self.assertAlmostEqual(parsed.pose.yaw, -0.0475, delta=0.01)
        self.assertGreater(sample["position"]["z"], 0.52)

    def test_recorded_pre_reset_fallen_sample_fails_m1_height_gate(self):
        with self.assertRaisesRegex(AdapterInputError, "below"):
            odometry_to_pose(make_odom(self.samples[1]), received_steady_s=1.0)

    def test_odometry_message_stamp_must_advance(self):
        sample = self.samples[0]
        stamp_ns = sample["header"]["sec"] * 1_000_000_000 + sample["header"]["nanosec"]
        with self.assertRaisesRegex(AdapterInputError, "repeated or moved backwards"):
            odometry_to_pose(
                make_odom(sample), received_steady_s=1.0, last_stamp_ns=stamp_ns
            )


class MessageContractTests(unittest.TestCase):
    def test_odometry_requires_expected_parent_and_child_frames(self):
        sample = json.loads(FIXTURE.read_text(encoding="utf-8"))["samples"][0]
        message = make_odom(sample)
        message.header.frame_id = "map"
        with self.assertRaisesRegex(AdapterInputError, "expected odom"):
            odometry_to_pose(message, received_steady_s=1.0)

        message = make_odom(sample)
        message.child_frame_id = "base_footprint"
        with self.assertRaisesRegex(AdapterInputError, "expected odom"):
            odometry_to_pose(message, received_steady_s=1.0)

    def test_odometry_rejects_zero_quaternion_and_invalid_stamp(self):
        sample = json.loads(FIXTURE.read_text(encoding="utf-8"))["samples"][0]
        message = make_odom(sample)
        message.pose.pose.orientation = NS(x=0.0, y=0.0, z=0.0, w=0.0)
        with self.assertRaisesRegex(AdapterInputError, "norm is zero"):
            odometry_to_pose(message, received_steady_s=1.0)

        message = make_odom(sample)
        message.header.stamp.nanosec = 1_000_000_000
        with self.assertRaisesRegex(AdapterInputError, "out of range"):
            odometry_to_pose(message, received_steady_s=1.0)

    def test_pose_stamped_goal_requires_exact_odom_frame(self):
        goal = pose_stamped_to_goal(make_goal())
        self.assertEqual(goal.frame_id, "odom")
        self.assertEqual(goal.x, 0.5)
        with self.assertRaisesRegex(AdapterInputError, "required frame"):
            pose_stamped_to_goal(make_goal(frame="map"))

    def test_pose_stamped_goal_rejects_nonplanar_orientation(self):
        with self.assertRaisesRegex(AdapterInputError, "excessive"):
            pose_stamped_to_goal(
                make_goal(q=(math.sin(0.2), 0.0, 0.0, math.cos(0.2)))
            )

    def test_proposal_output_cannot_target_live_mc_topics(self):
        self.assertEqual(
            DEFAULT_PROPOSAL_TOPIC,
            "/robotsim/navigation/cmd_vel_proposal",
        )
        validate_proposal_topic(DEFAULT_PROPOSAL_TOPIC)
        for topic in ("/cmd_vel", "/aima/mc/locomotion/velocity"):
            with self.subTest(topic=topic), self.assertRaises(AdapterInputError):
                validate_proposal_topic(topic)

    def test_odometry_and_pose_stamped_fields_drive_bounded_command(self):
        sample = json.loads(FIXTURE.read_text(encoding="utf-8"))["samples"][0]
        odom = odometry_to_pose(make_odom(sample), received_steady_s=1.0)
        goal = pose_stamped_to_goal(make_goal(x=1.0, y=-0.05))
        controller = PointGoalController(ControllerConfig())

        output = controller.step(odom.pose, goal, now_steady_s=1.0)

        self.assertEqual(output.state, NavState.DRIVE)
        self.assertGreater(output.linear_x, 0.0)
        self.assertLessEqual(output.linear_x, 0.20)
        self.assertLessEqual(abs(output.angular_z), 0.30)


try:
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    ROS_MESSAGES_AVAILABLE = True
except ImportError:
    ROS_MESSAGES_AVAILABLE = False


@unittest.skipUnless(ROS_MESSAGES_AVAILABLE, "ROS 2 message packages are not installed")
class OptionalActualRosMessageTests(unittest.TestCase):
    def test_actual_odometry_and_pose_stamped_objects(self):
        odom = Odometry()
        odom.header.stamp = Time(sec=100, nanosec=1)
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.z = 0.65
        odom.pose.pose.orientation.w = 1.0
        parsed = odometry_to_pose(odom, received_steady_s=1.0)
        self.assertAlmostEqual(parsed.pose.yaw, 0.0)

        goal_msg = PoseStamped()
        goal_msg.header.stamp = Time(sec=100, nanosec=2)
        goal_msg.header.frame_id = "odom"
        goal_msg.pose.position.x = 1.0
        goal_msg.pose.orientation.w = 1.0
        goal = pose_stamped_to_goal(goal_msg)
        self.assertEqual(goal.x, 1.0)


if __name__ == "__main__":
    unittest.main()
