# MuJoCo Backend

This directory contains RobotSim's MuJoCo physics backend.

## Responsibilities

MuJoCo is the physics source of truth for:

- robot dynamics
- joints and actuators
- contacts and collision response
- physical object state
- internal robot state
- simulation ground truth

## Scope

This directory may contain:

- RobotSim-owned MuJoCo scenes
- model overlays and adapters
- task objects and environment assets
- physics configuration
- synchronization code
- backend utilities

Upstream vendor robot models should not be edited in place. RobotSim-specific changes should be implemented through overlays, generated files, adapters, or clearly separated derived assets.

## Unity Relationship

Unity is used as the high-fidelity scene, rendering, and external sensor frontend.

Unity must not become an independent second source of robot physics.

When geometry is represented in both MuJoCo and Unity, the project must define how the representations are synchronized and which representation is authoritative.

## ROS 2 Relationship

MuJoCo state and commands should be exposed through RobotSim's ROS 2 interfaces where appropriate.

Simulation ground truth must remain separate from estimated state used by localization, mapping, perception, and navigation algorithms.

## Versioning

The MuJoCo version used by RobotSim must be pinned and documented.

When the embedded MuJoCo Unity plugin is used, the native MuJoCo library and Unity plugin code must use compatible versions.

## Headless G1 smoke test

Run `./scripts/run_g1_mujoco_smoke.sh` from the repository root to execute the
project-owned headless model and stepping test in a fresh Python 3.12 virtual
environment. The runner fetches the locked `unitree_mujoco` source into a
temporary directory, installs the pinned MuJoCo Python binding, tests both
official G1 MJCF files, and removes its temporary checkout and environment when
it exits. The MJCF and meshes stay in the ignored upstream checkout; RobotSim
does not redistribute those assets.

The tested MuJoCo release is 3.3.6 and both source files come from the locked
`unitree_mujoco` commit. The compiled dimensions are `nq=36`, `nv=35`, `nu=29`,
`njnt=30`, `nbody=31`, and `nsensordata=113` for both variants. The 23-DOF
variant compiles to 62 geoms; the 29-DOF variant compiles to 73. These are the
dimensions of the upstream MJCF as loaded, including its free root and any
placeholder joints.

For each model the test checks key leg, waist, arm, and free-root joint names,
then advances 50 fixed steps at the MJCF timestep of 0.002 s. A requested
`+1.0 N·m` command on `left_shoulder_pitch` is clamped to `+0.5 N·m` and applied
to `left_shoulder_pitch_joint`. The same initial state is also stepped with zero
input; the test requires the commanded joint to move in the positive direction
by more than `1e-3 rad`, checks relevant state and sensor arrays for finite
values at every step, and replays the command to compare final joint, actuator,
acceleration, sensor, and control state within `1e-12`.

The test uses no random seed or renderer. It records the MuJoCo Python and
native versions, source commit, variant, compiled dimensions, timestep, fixed
step count, requested and applied command, joint result, and replay error in
its `G1_MUJOCO_SMOKE` output. The replay check establishes repeatability for
the same build and process; it does not claim bitwise agreement across
operating systems or CPU architectures. This smoke test validates loading,
actuation, and finite stepping only; it does not validate balance or walking.
