# robotsim_navigation - Issue 80 research adapter

This ament_python package wraps the existing planar point-goal controller in ROS 2. It consumes actual `nav_msgs/msg/Odometry` and `geometry_msgs/msg/PoseStamped` message objects, and publishes `geometry_msgs/msg/Twist` proposals. It accepts only odometry with `header.frame_id=odom` and `child_frame_id=base_link`; goals must already be expressed in `odom`. No `map -> odom` transform is guessed and no tf2 lookup is performed.

The adapter rejects non-monotonic/zero odometry stamps, invalid or non-finite poses, unsafe base height (`z < 0.52 m` by default), roll/pitch above 15 degrees, wrong frames, and stale received odometry. Staleness uses local monotonic receive age; it does not compare ROS/simulation stamp to wall time because no clock mapping has been established. A backwards odometry timestamp requires restarting the node to begin a new session.

Goal input is `PoseStamped` on `/robotsim/navigation/goal`. Cancellation is `std_msgs/msg/Empty` on `/robotsim/navigation/cancel`. The adapter enforces controller goal timeout and arrival dwell, rejects a new goal while one is active or during the post-stop hold, publishes zero `Twist` on cancellation, timeout, stale/invalid odometry, terminal state, and continuously while idle. The timer uses a steady clock.

The only Twist output is the fixed, isolated proposal topic `/robotsim/navigation/cmd_vel_proposal`. The node has no MC client, source registration, mode switch, or subscription to a live controller. It checks ROS name resolution and rejects remapping the output to `/cmd_vel` or `/aima/mc/locomotion/velocity`. Keep it disconnected from live MC until Issue 80 M1 passes and a separate integration change is reviewed.

## Offline checks

From the repository root, with Python 3:

```sh
PYTHONPATH=ros2_ws/src/robotsim_navigation python3 -B -m unittest discover -s scripts/research/issue80 -p 'test_*.py' -v
PYTHONPATH=ros2_ws/src/robotsim_navigation python3 -B -m unittest discover -s ros2_ws/src/robotsim_navigation/test -p 'test_*.py' -v
```

The tests use ROS-message-shaped synthetic objects and two preserved M1 Odometry samples. Construction with actual generated ROS message classes is an optional test and is skipped if the ROS message packages are unavailable. These checks are not a ROS graph test or physical navigation acceptance.

For a real ROS 2 Humble message-class and graph check in the existing isolated runtime image, run `bash scripts/research/issue80/run_adapter_ros_validation.sh` from the worktree root. The script verifies the local image content ID and uses Docker `--pull=never`, `--network none`, a private `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY=1`, a read-only source mount and a disposable `/tmp` workspace. The graph test refuses to run unless `ROBOTSIM_ISOLATED_ROS_GRAPH=1` is set. It launches the installed `ros2 run robotsim_navigation point_goal_adapter` entry point and validates proposal publication, safety stops, and rejection of remaps to live command topics; it does not connect to MC.

## ROS 2 Humble build and isolated proposal run

On a machine that already has ROS 2 Humble and `colcon` installed (none were installed for this task):

```sh
source /opt/ros/humble/setup.bash
cd <RobotSim-checkout>/ros2_ws
colcon build --packages-select robotsim_navigation
source install/setup.bash
ros2 run robotsim_navigation point_goal_adapter
```

Defaults are `/aima/hal/odom/state`, `/robotsim/navigation/goal`, and `/robotsim/navigation/cancel`; proposals go only to `/robotsim/navigation/cmd_vel_proposal`. Do not remap that proposal output to MC.
