# Issue #80 ROS 2 point-goal adapter validation

## Scope

This validates the standalone ROS 2 adapter and the existing point-goal controller. It does not connect to Local03, MC, or MuJoCo and does not claim physical navigation.

The adapter consumes actual `nav_msgs/msg/Odometry` and `geometry_msgs/msg/PoseStamped` objects, requiring Odometry `header.frame_id=odom`, `child_frame_id=base_link`, and goal frame `odom`. It publishes `geometry_msgs/msg/Twist` only to `/robotsim/navigation/cmd_vel_proposal`. It has no MC clients or velocity-topic output. ROS 2 Humble defines Odometry pose in `header.frame_id`, twist in `child_frame_id`, and PoseStamped as a stamped pose: [Odometry.msg](https://github.com/ros2/common_interfaces/blob/humble/nav_msgs/msg/Odometry.msg), [PoseStamped.msg](https://github.com/ros2/common_interfaces/blob/humble/geometry_msgs/msg/PoseStamped.msg).

## Environment and build

- WSL host has no ROS 2, colcon, `rclpy`, `nav_msgs`, or `geometry_msgs`; nothing was installed on the host.
- Existing image `robotsim:humble-raicom-runtime-20261019`, image ID `sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31`, provided ROS 2 Humble, Python 3.10.12, colcon, `ament_python`, `rclpy`, `nav_msgs`, `geometry_msgs`, `std_msgs`, and generated ROS message classes.
- `colcon build --packages-select robotsim_navigation`: PASS.
- `python3 -m pytest -q .../test --junitxml=...`: 11 passed, 0 failed, 0 skipped. The actual generated ROS-message object test ran and passed.
- The package test runner also completed all 11 tests successfully under `colcon test`.
- ROS remapping the proposal to either `/cmd_vel` or `/aima/mc/locomotion/velocity` was tested against Humble; the adapter process exits during startup before creating the proposal publisher.

The container ran with `--network none`, a private `ROS_DOMAIN_ID=194`, `ROS_LOCALHOST_ONLY=1`, a read-only package source mount, and an ephemeral `/tmp` workspace. It was removed at exit. The test harness requires `ROBOTSIM_ISOLATED_ROS_GRAPH=1`, preventing accidental execution against a workstation ROS graph.

## ROS graph results

The test launched the installed entry point using `ros2 run robotsim_navigation point_goal_adapter` and used generated message classes for all publishers/subscribers.

- Actual Odometry subscription and PoseStamped goal reception produced a positive bounded proposal on `/robotsim/navigation/cmd_vel_proposal` (limits observed: linear x <= 0.20 m/s, angular z <= 0.30 rad/s).
- Wrong goal frame (`map`) was rejected; the idle proposal remained zero.
- Wrong Odometry parent frame and repeated Odometry stamp stopped an active proposal with zero Twist.
- Unsafe base height and unsafe roll were rejected and stopped the active proposal with zero Twist.
- `std_msgs/msg/Empty` cancellation stopped an active proposal with zero Twist.
- Halting Odometry delivery triggered stale-reception rejection and zero Twist.
- A goal timeout while fresh Odometry continued stopped the proposal with zero Twist.
- After cancellation, the graph observed repeated zero Twist messages while idle, not only the single immediate stop sample.
- ROS graph introspection found the proposal output and status topic. It found no adapter publisher on `/cmd_vel` or `/aima/mc/locomotion/velocity`, and no adapter service under MC/locomotion namespaces.

Odometry freshness uses monotonic receive age plus rejection of zero/repeated/backward stamps among accepted samples. It does not compare ROS/simulation stamp age with wall time because no clock mapping is established. `PoseStamped.header.stamp` is range-validated, but it is not used for age rejection or a transform lookup; goals are accepted only when already in `odom`. These are explicit clock/frame boundaries. The adapter must remain disconnected from live MC until M1 passes and a separate integration change is reviewed.

QoS is bounded and transient-state oriented: Odometry subscription is depth 1, best-effort, volatile; proposal publisher is depth 1, reliable, volatile; goal/cancel use reliable volatile depth 10. The Humble graph test successfully delivered Odometry from a reliable publisher to the best-effort subscription and observed all commands on the reliable proposal publisher. The live MC Odometry endpoint was not connected, so its negotiated QoS was not tested here.

The runner now verifies the local Docker image ID `sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31` and runs with `--pull=never`; a missing or retagged image fails before execution instead of silently substituting a different image.

## Earlier offline evidence retained

The pure controller suite remains 11/11 passing. The host-only adapter field suite is 9 passed and 1 actual-message test skipped because ROS packages are absent on the host. The graph test is also skipped on the host unless the explicit isolated-graph flag is set. In the Humble container, the actual-message test and graph test both passed. The two recorded M1 Odometry fixtures retain their sample values and source hashes. Their authoritative sources are `reviews/issue-80/x2-nav-m1-20261019T125515Z/logs/post-reset-odom.yaml` (SHA-256 `dc5a8579719c706b098e3193e903c50cd439314c8adb790cf85fecc0466760dc`) and `reviews/issue-80/x2-nav-m1-corrected-gate-20261019T153421Z/pre_reset_odom.yaml` (SHA-256 `5d6636f9769c706b098e3193e903c50cd439314c8adb790cf85fecc0466760dc`). The package fixture keeps only stable run IDs and filenames; local evidence paths stay in this research report.

## Reproduction

From the worktree root, with the existing Docker image available:

```sh
bash scripts/research/issue80/run_adapter_ros_validation.sh
```

The script builds the package in an isolated temporary workspace, launches the adapter in a private ROS graph, exercises real ROS messages and safety stops, and removes the container. This is ROS adapter validation only, not locomotion or physical navigation acceptance.
