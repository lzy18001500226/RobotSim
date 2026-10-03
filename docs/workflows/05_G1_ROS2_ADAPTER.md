# Workflow 05: G1 ROS 2 Adapter

## Scope

This workflow builds the initial topology-independent RobotSim ROS 2 boundary
and the Unitree G1 SDK2 adapter. It targets the pinned G1 MuJoCo SDK2 simulator
on an isolated local DDS domain. It does not provide root pose/twist, real
hardware support, navigation, perception, locomotion policy integration, or
Unity synchronization. The SDK2 `LowState` carries joint and IMU state; its
separate high-state stream has no shared tick or sample identifier, so root
state is explicitly unavailable instead of being fused without alignment.

## Packages and semantics

`ros2_ws/src/robotsim_interfaces` defines `RobotState` and `RobotCommand`.
`ros2_ws/src/robotsim_core` owns the state generation/sequence timeline and a
single latest-value torque command slot. G1-only SDK2 code lives in
`robots/unitree_g1/ros2/robotsim_g1_adapter`.

The adapter maps the 29 G1 joints by the names and order in the pinned SDK2
`defines.h`. It uses `LowState.tick` as milliseconds of simulator time, which
matches the pinned `unitree_mujoco` bridge's `round(mj_data->time / 1e-3)`.
This is a simulator-specific mapping; real-device clock conversion is not
claimed. Time rollback starts a new opaque generation and resets state and
command sequences. A process restart creates a fresh generation token. The
ROS `/clock` and state stamps are sourced from the same tick.

The model identity is
`unitree_g1:unitree_mujoco@1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d:scene_29dof`.
The state origin is always marked `ground_truth` and is written only when the
node receives the explicit `state_origin:=simulator_ground_truth` setting.
Commands currently accept one source ID, torque commands, all 29 explicitly
named joints, finite values within a configured effort limit, the active
generation, and a sequence starting at zero and increasing strictly. The active command replaces
the previous value in a one-slot buffer. A steady-clock timeout clears it and
the adapter sends zero torque. The initial timeout (`250 ms`) and effort limit
(`0.2 Nm`) are local smoke-test defaults, not policy or safety limits for a
robot deployment.

Topics and QoS:

| Topic | Type | QoS | Meaning |
|---|---|---|---|
| `/simulation/ground_truth/g1/state` | `robotsim_interfaces/RobotState` | best effort, volatile, keep last 1 | Ground-truth joints and IMU; root pose/twist unavailable |
| `/clock` | `rosgraph_msgs/Clock` | ROS `ClockQoS` | Simulator time from `LowState.tick` |
| `/robot/g1/command` | `robotsim_interfaces/RobotCommand` | reliable, volatile, keep last 1 | One-source bounded torque setpoint |
| `/tf_static` | `tf2_msgs/TFMessage` | static broadcaster transient local | Identity `base_link` to `imu_link` transform |

`base_link` is the waist/root frame; body coordinates use REP-103. The only TF
edge emitted here is static `base_link -> imu_link`. `world -> base_link` is
not emitted because the available SDK2 streams do not provide aligned root
pose state.

## Build and tests

The commands below assume a Humble container with the pinned vendor sources
mounted read-only at `/vendor`, the worktree at `/workspace`, and temporary
build files at `/lab`:

```bash
source /opt/ros/humble/setup.bash
cmake -S /vendor/unitree_sdk2 -B /lab/sdk-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/lab/sdk-prefix \
  -DBUILD_EXAMPLES=OFF
cmake --build /lab/sdk-build -j2
cmake --install /lab/sdk-build

cd /workspace
colcon --log-base /lab/colcon-log build \
  --base-paths ros2_ws/src robots/unitree_g1/ros2 \
  --build-base /lab/colcon-build --install-base /lab/colcon-install \
  --cmake-args -DCMAKE_PREFIX_PATH=/lab/sdk-prefix
colcon --log-base /lab/colcon-log test \
  --base-paths ros2_ws/src robots/unitree_g1/ros2 \
  --build-base /lab/colcon-build --install-base /lab/colcon-install
colcon --log-base /lab/colcon-log test-result \
  --test-result-base /lab/colcon-build --verbose
```

## L2 headless model check

Run the existing MuJoCo smoke against the same pinned, clean vendor checkout.
The Python dependencies and test output stay under `/lab`:

```bash
python3 -m pip install --target /lab/l2-python-deps \
  -r /workspace/simulation/mujoco/requirements-smoke.txt
cd /tmp
git config --global --replace-all safe.directory /vendor/unitree_mujoco
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/lab/l2-python-deps \
ROBOTSIM_UNITREE_MUJOCO_DIR=/vendor/unitree_mujoco \
  python3 -m unittest discover \
    -s /workspace/simulation/mujoco/tests -v
```

## Isolated simulator run

Use a separate, writable overlay for the pinned simulator; keep `/vendor`
read-only. The simulation and ROS adapter must run in the same container so
SDK2 and ROS DDS traffic remains on the container's loopback interface.

```bash
mkdir -p /lab/sim-overlay
cp -a /vendor/unitree_mujoco /lab/sim-overlay/unitree_mujoco
ln -sfn /opt/mujoco/mujoco-3.3.6 \
  /lab/sim-overlay/unitree_mujoco/simulate/mujoco
cmake -S /lab/sim-overlay/unitree_mujoco/simulate \
  -B /lab/sim-overlay/unitree_mujoco/simulate/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/lab/sdk-prefix
cmake --build /lab/sim-overlay/unitree_mujoco/simulate/build -j2

export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1
export LD_LIBRARY_PATH=/lab/sdk-prefix/lib:/opt/mujoco/mujoco-3.3.6/lib:/opt/ros/humble/lib
cd /lab/sim-overlay/unitree_mujoco/simulate/build
./unitree_mujoco -r g1 -i 73 -n lo -s scene_29dof.xml -t 1
```

In another shell in that same container, with the simulator still running:

```bash
source /opt/ros/humble/setup.bash
source /lab/colcon-install/setup.bash
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=/lab/colcon-install/robotsim_interfaces/lib:/lab/sdk-prefix/lib:/opt/mujoco/mujoco-3.3.6/lib:/opt/ros/humble/lib:/opt/ros/humble/local/lib
ros2 run robotsim_g1_adapter g1_ros2_adapter --ros-args \
  -p state_origin:=simulator_ground_truth \
  -p command_timeout_ms:=250 -p max_effort_nm:=0.2
```

Run the live integration check from a third shell in the same container:

```bash
source /opt/ros/humble/setup.bash
source /lab/colcon-install/setup.bash
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=/lab/colcon-install/robotsim_interfaces/lib:/lab/sdk-prefix/lib:/opt/mujoco/mujoco-3.3.6/lib:/opt/ros/humble/lib:/opt/ros/humble/local/lib
python3 /workspace/robots/unitree_g1/ros2/robotsim_g1_adapter/test/live_roundtrip.py
```

The live check prints `restart_reference_generation`. To verify process
restart semantics, start the observer below while the adapter is still
running, stop the adapter with Ctrl-C in its shell, then relaunch the same
`ros2 run` command. The observer must report a different generation; the new
adapter log must report its first published sequence as zero.

```bash
python3 /workspace/robots/unitree_g1/ros2/robotsim_g1_adapter/test/restart_generation.py \
  <restart_reference_generation>
```

The pinned SDK2 library bundles Cyclone DDS. In the tested adapter process,
selecting ROS Humble's Cyclone RMW as well caused SDK2 explicit domain
initialization to fail. The local integration run therefore uses Fast DDS for
the ROS graph and SDK2's Cyclone DDS for its existing Unitree channels, both
restricted to domain 73 and loopback. This is a local compatibility setting;
the adapter code does not select an RMW or settle the project transport
decision.

The check validates simulator state publication, tick/stamp semantics,
generation filtering, topic QoS, the static frame, 20 bounded torque
round-trips, and timeout fallback. Its send-to-feedback latency is measured
from the ROS publisher's local monotonic send time to the first state callback
whose `tau_est` crosses the requested sign/magnitude. This is observed
round-trip timing, not simulator actuation latency or a real-time guarantee.
