# Unitree G1 Backend

This directory contains project-owned G1 adapter code and configuration.
The initial ROS 2 SDK2 adapter is in
[`ros2/robotsim_g1_adapter/`](ros2/robotsim_g1_adapter/) and its build and
simulator validation procedure is
[`docs/workflows/05_G1_ROS2_ADAPTER.md`](../../docs/workflows/05_G1_ROS2_ADAPTER.md).
The topology-independent interfaces and state/command semantics live under
[`ros2_ws/src/`](../../ros2_ws/src/).

Upstream Unitree repositories belong in `third_party/`, not here.
Exact stock and hand-equipped assembly identities are recorded in the
[robot asset registry](../../assets/robots/README.md). The registry does not
store vendor meshes or automatically approve an assembly.
