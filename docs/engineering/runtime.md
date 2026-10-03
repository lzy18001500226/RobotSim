# Runtime and Sim2Real contract

These conventions apply when runtime code is introduced. They preserve the current [MuJoCo, Unity, and ROS 2 boundary](../adr/0004_mujoco_unity_ros2_architecture.md); they do not claim that RobotSim is hard real-time or that every interface below already exists.

## Language by runtime characteristics

**RobotSim decision:** use C++17 as the default compiled runtime standard for ROS 2 Humble and Ubuntu 22.04. REP-2000 lists Ubuntu Jammy 22.04 as a Tier 1 Humble platform and C++17 as the minimum language standard. C++17 is sufficient for the current baseline; do not raise the workspace standard for convenience. Before adopting an SDK, pin it and build it with the target compiler and C++17. If an upstream SDK imposes a different requirement, isolate that boundary and document the compatibility decision before changing the workspace standard. The current `third_party/LOCK.md` still contains TODO entries, so vendor compatibility is not yet verified.

- Prefer C++ for robot hardware adapters, actuator command and state paths, control loops, high-rate transport, latency-sensitive perception, and physics/runtime hot paths. Use C where a microcontroller or lower-level ABI calls for it.
- Python is appropriate for research, offline analysis, data conversion, plotting, training, experiment orchestration, repository tooling, CI, and prototypes. Python is not banned from online algorithms: use it only when the component's update rate, latency, resource, and deployment budgets are explicit and measurements on the target show they are met. A prototype is not production runtime evidence.
- Model work may use Python/PyTorch for research and export to an optimized runtime representation where deployment needs it.
- C# belongs in Unity rendering, scene integration, external sensor front-end, editor tooling, and synchronization. It does not become the authority for robot dynamics or low-level control.

## Timing and performance

**RobotSim decision:** before a component enters a timing-sensitive path, record its intended update rate or trigger model, end-to-end latency budget, tolerated jitter, target hardware, and overload behavior. There is no universal RobotSim frequency or jitter threshold. Measure against the actual target and representative load; do not infer real-time behavior from source shape, average FPS, or a cloud container.

Keep fast control work bounded and separate from slow perception, planning, disk/network I/O, and visualization. In paths that require real-time behavior, avoid blocking I/O, unbounded waits, dynamic allocation/deallocation, and excessive logging. Use bounded communication between real-time and non-real-time work, and document what happens on queue overflow or a missed deadline. The Humble `ros2_control` controller manager owns a configured update loop for hardware read, controller update, and command write; `realtime_tools` documents why a normal ROS publisher is not suitable inside a real-time controller update and provides a non-real-time publishing handoff. Follow the project component's real timing model rather than copying a rate from a tutorial.

Asynchronous work is acceptable when ownership, cancellation, ordering, queue limits, and timestamps are explicit. ROS 2's default executor does not itself establish hard real-time scheduling guarantees. Claims such as “real-time”, “deterministic”, or a numeric latency bound require target-specific measurements and the method, workload, and environment recorded with the result.

## ROS and simulation interfaces

### Units and frames

**RobotSim decision adopting the REP-103 convention:** use SI units and right-handed body frames: x forward, y left, z up. Use radians for angles. Camera optical frames follow REP-103's optical convention (z forward, x right, y down); do not silently reuse body-frame axes. REP-103 permits justified and documented exceptions; state any RobotSim deviation in the affected interface contract.

**RobotSim convention:** use REP-105's `map`, `odom`, and robot base-frame semantics. `odom` is locally continuous and may drift; `map` is globally corrected and may jump. Localization owns `map -> odom`; odometry owns `odom -> base_link`. For humanoids, use REP-120 as the starting point: `base_link` is attached at the waist/root; a planar `base_footprint`, when useful, is derived from support feet rather than treated as a rigid link. A robot-specific deviation must be stated in its description and TF contract.

Every message with spatial values must carry the correct frame and unit. Publish TF edges from one declared authority each. Distinguish sensor frames, body frames, world frames, and optical frames in names and transforms.

### Time and state truth

Stamp sensor measurements at acquisition time and preserve that stamp through processing; a callback or publish timestamp is not a substitute. Use ROS time for simulation-coherent sensor/state messages. When a graph is running against simulation or bag playback, configure every participating node consistently with `use_sim_time` and provide one authoritative `/clock` source. ROS time may pause, jump, or move backward with simulation controls; use a steady/monotonic clock for watchdogs, host deadlines, and elapsed wall-time measurements. Handle the absence of an initial simulation clock explicitly.

MuJoCo state is simulation ground truth. Localization, perception, and state-estimation outputs are estimates, including when their input is simulated. Keep these channels, topic names, message ownership, and data products distinct; ground truth must never masquerade as `/odom`, a perception result, or a localization estimate. Record the source and timestamp semantics of derived state.

### Ownership, QoS, and backend parity

For each topic, service, and action introduced, document its owner, message or request meaning, frame/time semantics, and the expected publishers and consumers. Use topics for streams, services for bounded request/response operations, and actions for long-running operations that need feedback or cancellation. Select QoS per interface: state required reliability, history/depth, durability, and any deadline/liveliness needs. High-rate sensor data may favor best-effort delivery; commands and task actions may need reliable delivery and explicit freshness/safety handling. Do not copy one QoS profile to every interface.

Expose common, robot-independent interfaces above backend adapters. Keep Unitree and AgiBot SDK types, transport, and lifecycle details under `robots/unitree_g1/` and `robots/agibot_x2/`; do not make navigation, perception, manipulation, or task logic depend directly on a vendor SDK. Keep the same interface meaning across simulated and real backends when practical, and document where parity is impossible. MuJoCo remains the physical authority; Unity synchronizes its scene and external sensors to that authority rather than introducing a second robot dynamics world.

## Sources and status

- **Upstream convention references:** [REP-103](https://www.ros.org/reps/rep-0103.html), [REP-105](https://www.ros.org/reps/rep-0105.html), [REP-120](https://www.ros.org/reps/rep-0120.html), and [REP-2000](https://www.ros.org/reps/rep-2000.html). RobotSim's adopted rules above are project decisions; REP documents may permit justified, documented deviations.
- **Quality vocabulary, not a project claim:** [REP-2004](https://www.ros.org/reps/rep-2004.html) defines ROS package quality-level criteria. RobotSim does not claim a quality level; any future claim must be supported package-by-package with evidence.
- **Upstream Humble implementation guidance:** [ros2_control controller manager](https://control.ros.org/humble/doc/ros2_control/controller_manager/doc/userdoc.html), [`realtime_tools` publisher guidance](https://github.com/ros-controls/realtime_tools/blob/humble/doc/index.rst), [ROS 2 Humble real-time programming](https://docs.ros.org/en/humble/Tutorials/Demos/Real-Time-Programming.html), and [ROS 2 Humble executors](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html).
- **ROS time:** [ROS 2 Humble Clock and Time](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Clock-and-Time.html).
- **Upstream architecture patterns, not requirements:** [MoveIt 2 Humble `move_group`](https://moveit.picknik.ai/humble/doc/concepts/move_group.html), [motion planning](https://moveit.picknik.ai/humble/doc/concepts/motion_planning.html), and [planning scene monitor](https://moveit.picknik.ai/humble/doc/concepts/planning_scene_monitor.html) separate planning requests, world/robot state, and controller interfaces behind composable components. [Nav2 Humble core plugin interfaces](https://github.com/ros-navigation/navigation2/tree/humble/nav2_core) and its [planner server](https://github.com/ros-navigation/navigation2/blob/humble/nav2_planner/include/nav2_planner/planner_server.hpp) demonstrate lifecycle-managed task servers hosting algorithm plugins. Adopt a pattern only where it fits RobotSim's needs.
