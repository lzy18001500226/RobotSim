# Issue #80 offline controller evidence

The pure controller core now lives in the ROS 2 package at ros2_ws/src/robotsim_navigation/robotsim_navigation/point_goal_controller.py. The adapter imports the actual nav_msgs/Odometry, geometry_msgs/PoseStamped/Twist, and std_msgs/Empty/String types. Its default Twist output is an isolated proposal topic; it has no MC client.

Offline standard-library tests from the worktree root:
PYTHONPATH=ros2_ws/src/robotsim_navigation python3 -B -m unittest discover -s scripts/research/issue80 -p 'test_*.py' -v
PYTHONPATH=ros2_ws/src/robotsim_navigation python3 -B -m unittest discover -s ros2_ws/src/robotsim_navigation/test -p 'test_*.py' -v

Recorded sample tests use preserved M1 Odometry fields. These checks are synthetic/source-contract evidence, not ROS graph, MuJoCo, locomotion, or navigation acceptance.