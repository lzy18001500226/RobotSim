# RobotSim ROS 2 Workspace

This is the topology-independent ROS 2 workspace. It currently defines
`robotsim_interfaces` and `robotsim_core`; vendor-specific adapters remain
under `robots/<vendor_robot>/ros2/` and are included as additional colcon base
paths. See the [G1 adapter workflow](../docs/workflows/05_G1_ROS2_ADAPTER.md)
for build, isolated DDS, simulator, and integration commands.
