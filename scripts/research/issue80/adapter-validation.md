# Issue #80 ROS adapter offline validation

## Scope

The `robotsim_navigation` ament Python package wraps the existing point-goal controller with ROS 2 subscriptions for `nav_msgs/msg/Odometry` and `geometry_msgs/msg/PoseStamped`. Odometry must use `header.frame_id=odom` and `child_frame_id=base_link`; goals must already be in `odom`. The adapter produces `geometry_msgs/msg/Twist` only on `/robotsim/navigation/cmd_vel_proposal`. It has no MC service/client path and refuses the known live velocity topics.

ROS 2 Humble defines Odometry pose in `header.frame_id` and twist in `child_frame_id`; this adapter consumes the pose fields only. PoseStamped is a stamped pose, and the adapter validates its frame and planar pose without inventing a transform. References: [Odometry.msg](https://github.com/ros2/common_interfaces/blob/humble/nav_msgs/msg/Odometry.msg), [PoseStamped.msg](https://github.com/ros2/common_interfaces/blob/humble/geometry_msgs/msg/PoseStamped.msg).

## Offline results

- Pure controller suite: 11/11 passed, including synthetic differential-drive pose-goal convergence, bounded command generation, cancellation, goal timeout, stale-pose stop, frame mismatch, and zero-output terminal states.
- Adapter field-contract suite: 9 passed, 1 skipped (10 discovered). It covers recorded post-reset upright and pre-reset fallen odometry, monotonic message stamps, parent/child frame checks, quaternion/stamp validation, strict PoseStamped frame and planar orientation checks, bounded command generation from parsed Odometry/PoseStamped fields, and protected output topic selection.
- The skipped case constructs actual generated ROS message objects. `rclpy`, `nav_msgs`, `geometry_msgs`, `std_msgs`, `std_srvs`, `tf2_ros`, `ros2`, and `colcon` are absent in this WSL environment. No packages were installed. The node import, colcon build, ROS graph behavior, and actual generated-message test remain deferred.
- Python syntax compilation, package XML parsing, and `git diff --check` passed.
- No MuJoCo, MC, Docker, DDS, GUI, or physical navigation process was started for this task.

## Data and limits

The recorded fixture preserves two existing M1 samples and their source file SHA-256 values: `post_reset_upright` (accepted by the upright gate) and `pre_reset_fallen` (rejected by the base-height gate). Staleness is detected by local monotonic receive age; ROS/simulation stamp age is not compared with wall time because no clock mapping is established. Backward/repeated Odometry stamps fail closed and require restarting the adapter to establish a new session.

The node uses a steady-clock timer, controller goal timeout, cancellation topic, post-stop goal rejection window, zero Twist on active stop conditions, and continuous zero Twist while idle. The proposal output must remain disconnected from live MC until Issue #80 M1 passes and that integration receives a separate review.

## Reproduction

Run from the repository root:

```sh
PYTHONPATH=ros2_ws/src/robotsim_navigation python3 -B -m unittest discover -s scripts/research/issue80 -p 'test_*.py' -v
PYTHONPATH=ros2_ws/src/robotsim_navigation python3 -B -m unittest discover -s ros2_ws/src/robotsim_navigation/test -p 'test_*.py' -v
PYTHONPYCACHEPREFIX=/tmp/robotsim-issue80-pycache python3 -m py_compile ros2_ws/src/robotsim_navigation/robotsim_navigation/*.py scripts/research/issue80/*.py
python3 -c 'import xml.etree.ElementTree as ET; ET.parse("ros2_ws/src/robotsim_navigation/package.xml")'
git diff --check
```

These results validate the offline controller and message-field adapter only. They do not validate locomotion, localization, obstacle avoidance, the ROS graph, or physical navigation. Issue #80 M1 remains blocked; the adapter is not connected to MC.
