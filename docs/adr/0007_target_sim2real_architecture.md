# ADR-0007: Target simulation, control, sensor, and Sim2Real architecture

Status: Proposed

## Context

ADR-0004 already establishes the accepted high-level responsibility split:

- MuJoCo is the physics source of truth.
- Unity is the high-fidelity scene, rendering, and external-sensor frontend.
- ROS 2 is the system-integration layer.

This record refines that accepted boundary into a longer-term target architecture for:

- Unitree G1 and AgiBot X2 humanoids;
- future multirotor/UAV support;
- reinforcement learning, whole-body control, manipulation, perception, localization, and navigation;
- direct progression from simulation toward hardware without making Gazebo a required production dependency.

This proposal must not be treated as accepted proof that the selected process topology, transport, sensor implementation, or multi-robot performance is adequate. Issues #10, #11, #13, #14, and #15 provide the evidence needed to accept or revise the relevant parts.

## Proposed decision

RobotSim should evolve around four independently replaceable boundaries:

1. **Physics backend** — MuJoCo is the default physical authority.
2. **Sensor/render backend** — Unity is the current preferred implementation, behind a backend boundary rather than embedded as the definition of RobotSim.
3. **Robot control backend** — humanoid controllers and UAV/PX4 integrations map through robot-owned adapters.
4. **System integration** — ROS 2 connects perception, localization, navigation, task logic, and real-robot interfaces without being forced into every internal high-rate physics path.

The architecture should preserve one simulation clock/state contract across all of these boundaries.

## Target topology

```text
                       RobotSim
                          |
                Simulation clock/state
                          |
             +------------+-------------+
             |                          |
       Physics backend            Sensor/render backend
           MuJoCo                     Unity
             |                          |
   joints/contact/forces        RGB/depth/LiDAR/
   rigid-body dynamics          segmentation/rendering
             |                          |
             +------------+-------------+
                          |
                   Robot interfaces
                  /                \
             Humanoids             UAV
          G1 / X2 adapters     direct / PX4 SITL
                  \                /
                          |
                         ROS 2
                          |
        perception / localization / navigation /
                 manipulation / task logic
```

## Physics authority

MuJoCo owns:

- rigid-body dynamics;
- joints and actuators;
- contact and collision response;
- robot/object physical state;
- articulation state for physically interactive fixtures;
- simulation ground truth;
- physics-derived proprioception.

Unity must not become a second authority for robot or object dynamics.

High-rate physics/control loops should remain close to the MuJoCo backend rather than being forced through ROS 2 or the render process.

## Sensor ownership

### Physics-derived sensing

Prefer MuJoCo for signals that derive directly from the physical state:

- joint position/velocity;
- actuator state;
- IMU truth and derived inertial measurements;
- contact;
- force/torque;
- base state and simulation truth.

### Exteroceptive sensing

Prefer the sensor/render backend for:

- RGB;
- depth;
- segmentation;
- optical flow where needed;
- LiDAR / point cloud.

Unity is the first implementation target, but RobotSim should not make the state/control architecture depend on Unity-specific APIs. A future UE, Isaac, or headless sensor backend should be possible without replacing MuJoCo or robot-control code.

## Scene representation

RobotSim should avoid manually maintaining unrelated duplicate worlds.

The target is a canonical scene description or generation path that produces:

- simplified collision/articulation geometry for MuJoCo;
- high-fidelity visual/sensor geometry for Unity.

The two representations must share identifiers, transforms, units, articulation state, and simulation timestamps.

Examples:

- a refrigerator may be a small set of MuJoCo collision bodies plus a door hinge and handle collision;
- Unity may hold the detailed refrigerator mesh/materials while consuming the same door-joint state;
- a table or wall seen by LiDAR must not exist only in Unity while being absent from the physical scene unless explicitly documented as a non-physical visual object.

The exact canonical scene format is intentionally not selected yet.

## Timing and synchronization

The architecture requires explicit simulation time rather than relying on wall-clock coincidence.

Every exported state/sensor sample should be attributable to a simulation time/generation/sequence.

The following may run at different rates:

- MuJoCo physics;
- low-level control;
- Unity rendering;
- RGB-D capture;
- LiDAR acquisition;
- ROS 2 publication;
- perception/localization.

Different rates are acceptable; ambiguous timestamps are not.

Particular care is required for scanning sensors. A LiDAR scan may contain points acquired across a time interval rather than at one instantaneous render timestamp. Motion-distortion semantics must be defined before claiming FAST-LIO/VIO/localization fidelity.

## Humanoid control boundary

Humanoid algorithms may include:

- RL locomotion policy;
- whole-body control;
- IK;
- manipulation planner/controller.

The preferred boundary toward simulation and hardware is an explicit actuator/state contract such as:

```text
state:
  q / dq / base / contact / timing

command:
  q_des / dq_des / kp / kd / tau_ff
  or an explicitly documented lower-level torque command
```

Robot-specific details remain inside G1/X2 adapters.

The long-term Sim2Real goal is:

```text
same controller
     |
common command/state semantics
   /                     \
MuJoCo adapter       hardware/vendor adapter
```

A universal humanoid firmware/runtime should not be designed before working G1 and X2 examples reveal a real shared boundary.

## UAV control boundary

RobotSim should support two UAV modes rather than choosing only one.

### Direct research mode

```text
research controller
      |
rotor thrust / speed / actuator command
      |
    MuJoCo
```

Use this for controller research, MPC, RL, formation, safety filters, and fast headless experiments.

### PX4 firmware-in-the-loop mode

```text
autonomy / high-level command
            |
         PX4 SITL
            |
      actuator outputs
            |
     RobotSim PX4 bridge
            |
    rotor/motor dynamics
            |
          MuJoCo
```

RobotSim supplies simulated sensors to PX4 and applies PX4 actuator outputs to the MuJoCo vehicle.

This keeps PX4 estimator, control, allocation, failsafe, and firmware behavior in the loop when those are part of the validation target.

PX4 integration must not make Gazebo a required physical backend.

## UAV dynamics

MuJoCo being the physics backend does not automatically provide a high-fidelity multirotor model.

RobotSim will need explicit UAV models for at least:

- motor/rotor response;
- thrust and reaction torque;
- body drag;
- actuator saturation and delay.

Battery effects, ground effect, propwash, aerodynamic coupling, and higher-fidelity effects should be added only when hardware evidence or target experiments justify them.

## ROS 2 boundary

ROS 2 remains the integration layer for:

- perception;
- localization/mapping;
- navigation;
- manipulation/task-level integration;
- simulated/real robot interchange;
- external tooling.

ROS 2 should not be inserted into every internal high-rate physics operation merely for architectural uniformity.

Use bounded, timestamped interfaces between high-rate backend work and slower ROS-facing components.

## Gazebo role

The target architecture does **not** require Gazebo in the main RobotSim development path.

If RobotSim validates the required physics, sensors, timing, control interfaces, and PX4 bridge, normal development may proceed entirely through:

```text
MuJoCo + sensor/render backend + ROS 2 + robot/PX4 adapters
```

Gazebo may remain useful as an optional reference/regression backend, especially for comparing PX4 behavior against a mature official integration.

Gazebo is not intended to become:

- a second required production simulator;
- a mandatory step between RobotSim and hardware;
- a duplicate world/sensor implementation that RobotSim must maintain continuously.

## Sim2Real path

The target progression is:

```text
RobotSim simulation
      |
controller/interface parity
      |
HITL or protected/limited hardware
      |
full real hardware
```

For UAVs this may mean:

```text
RobotSim + PX4 SITL
      ->
PX4 HITL
      ->
prop-off / tethered / low-risk tests
      ->
full flight
```

For humanoids this may mean:

```text
MuJoCo
      ->
same controller through hardware adapter
      ->
supported/fixed-base hand or arm tests
      ->
protected standing / bounded motion
      ->
full task
```

A second simulator is not inherently a Sim2Real stage. Cross-simulator comparison is useful as regression evidence, not as a substitute for hardware-in-the-loop or protected hardware validation.

## Evidence gates before locking the architecture

The proposal becomes stronger only when the following gates have evidence.

### Gate 1 — robot actuation/control

Demonstrate representative G1 and X2 control/manipulation/locomotion paths in MuJoCo with explicit actuator semantics.

### Gate 2 — clock/state contract

Resolve Issues #10/#11 sufficiently to define:

- simulation clock;
- lifecycle/reset/step semantics;
- state/command identity;
- frames/units;
- sequence/generation handling;
- transport/performance requirements.

### Gate 3 — Unity state frontend

Issue #13 must demonstrate consistent state reproduction, interpolation, stale/reordered handling, and representative latency/frame-time behavior.

### Gate 4 — sensor fidelity

Issues #14/#15 must measure:

- geometry/depth/range accuracy;
- intrinsics/extrinsics;
- frame semantics;
- acquisition timestamp;
- rate;
- latency/jitter;
- queue/backpressure behavior;
- CPU/GPU cost;
- representative multi-robot scaling.

### Gate 5 — real-interface parity

Demonstrate that simulated and real adapters preserve sufficiently compatible controller/state semantics for at least one humanoid path, and later the PX4/UAV path.

### Gate 6 — hardware validation

Progress through HITL or protected/limited hardware before claiming Sim2Real readiness.

## Multi-robot performance rule

Do not infer scalability from programming language or average FPS.

Measure representative workloads using:

- physics update rate;
- controller rate;
- camera resolution/rate/count;
- LiDAR point rate/count;
- GPU/CPU utilization;
- GPU-to-CPU readback;
- allocations/copies;
- queue depth;
- end-to-end latency;
- jitter;
- overload behavior.

Prefer asynchronous/batched GPU sensor work and bounded buffers over per-ray/per-frame blocking CPU pipelines.

## Work deliberately avoided for now

Do not spend current project effort on:

- a Gazebo production backend;
- Unity robot physics;
- a UE sensor backend;
- an Isaac sensor backend;
- duplicated G1/X2 sensor implementations;
- a universal scene format before concrete scene synchronization requires one;
- a universal humanoid firmware/runtime before two robot backends establish the shared contract;
- exhaustive real-sensor noise models before baseline geometry/timing is correct;
- UAV/PX4 integration before the humanoid/core architecture gates provide a stable state/time boundary.

## Main risks

The highest architecture risks are not whether MuJoCo can step rigid-body dynamics.

They are:

1. Unity sensor fidelity for RGB-D/LiDAR;
2. MuJoCo-to-Unity timestamp/state synchronization;
3. multi-robot sensor throughput and backpressure;
4. keeping physical and visual scene geometry consistent;
5. simulated actuator/sensor semantics diverging from real hardware;
6. UAV motor/aerodynamic model fidelity when UAV support is added.

These should drive benchmark/evidence work.

## Relationship to existing records

- ADR-0004 remains the accepted high-level authority split.
- This ADR is a proposed refinement and future target.
- The topology/transport evidence from Issues #10/#11 may revise this proposal.
- Sensor evidence from #13/#14/#15 may cause Unity-specific choices to change without requiring the MuJoCo/control architecture to be replaced.

## Acceptance rule

Do not change this record to `Accepted` merely because the architecture is conceptually attractive.

Acceptance requires the maintainer to review evidence from the relevant architecture, timing, sensor, and performance gates and explicitly approve the resulting boundary.
