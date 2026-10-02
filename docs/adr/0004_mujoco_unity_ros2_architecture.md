# ADR-0004: MuJoCo physics, Unity frontend, ROS 2 integration

Status: Accepted

## Decision

RobotSim uses:

- MuJoCo as the physics source of truth
- Unity as the high-fidelity scene, rendering, and external sensor frontend
- ROS 2 as the system-integration layer

## MuJoCo Responsibilities

MuJoCo owns:

- robot dynamics
- joints and actuators
- contacts
- collision response
- physical state
- internal and proprioceptive state
- simulation ground truth

MuJoCo is the authoritative source for robot motion and physical interaction.

## Unity Responsibilities

Unity owns:

- high-fidelity visual environments
- RGB rendering
- depth rendering
- external sensor simulation
- visualization
- scene interaction and editing

Unity must not become an independent second physics source for the robot.

Where Unity-side sensor simulation requires scene geometry, that geometry must remain synchronized with the physical scene instead of being maintained as an unrelated second world.

## ROS 2 Responsibilities

ROS 2 connects simulation components with:

- perception
- localization
- mapping
- navigation
- manipulation
- task logic
- real robot backends

Simulated and real backends should preserve compatible high-level interfaces whenever practical.

## Timing

Physics stepping, rendering, and ROS 2 publication do not need to run at the same frequency.

High-frequency low-level control should remain close to the physics backend when practical.

Unity may render at a lower rate than MuJoCo physics.

ROS 2 sensor and state topics should use explicit timestamps and frame conventions.

## Consequences

- MuJoCo remains the single physics authority.
- Unity can evolve independently as the visualization and sensor frontend.
- ROS 2 remains representative of real robot deployment.
- Geometry synchronization between MuJoCo and Unity must be explicitly managed.
- Timing synchronization between physics, rendering, and ROS 2 must be explicitly managed.
- Simulation truth must remain separate from estimated state.
- If the embedded MuJoCo Unity plugin proves unstable under WSL, the architecture may use a separate MuJoCo process and Unity frontend without changing the high-level responsibility split.
