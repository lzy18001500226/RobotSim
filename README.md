# RobotSim

RobotSim is a simulation-first robotics platform built around MuJoCo, Unity, and ROS 2.

The goal is to provide a reusable simulation and system-integration framework for multiple robots, sensors, environments, and tasks, while keeping simulation interfaces close to real robot deployment.

## Architecture

- MuJoCo: robot dynamics, contacts, physics simulation, and ground truth
- Unity: high-fidelity scenes, rendering, external sensor simulation, and visualization
- ROS 2: perception, localization, navigation, manipulation, and task orchestration
- Robot adapters: robot-specific interfaces for platforms such as Unitree G1 and AgiBot X2

## Initial development targets

1. Establish a reproducible MuJoCo + ROS 2 simulation baseline.
2. Integrate Unity as the high-fidelity scene and sensor frontend.
3. Bring up Unitree G1 as the first robot backend.
4. Implement RGB-D perception and LiDAR localization, mapping, and navigation.
5. Implement manipulation tasks such as object grasping and fetching.
6. Add AgiBot X2 and generalize shared robot interfaces.
7. Keep simulation and real-robot ROS 2 interfaces aligned for Sim2Real deployment.

## Platform baseline

- Host: Windows 11
- Linux environment: WSL2 Ubuntu 22.04
- Container runtime: Docker
- ROS: ROS 2 Humble
- Physics engine: MuJoCo
- High-fidelity frontend: Unity
- Initial robot backends: Unitree G1 and AgiBot X2

## Repository structure

- apps/unity: Unity frontend and sensor simulation
- simulation/mujoco: MuJoCo physics backend
- robots: robot models and adapters
- ros2_ws: ROS 2 workspace
- docker: development and runtime containers
- data: datasets and generated data
- docs: architecture, setup, and workflows
- scripts: project utilities
- reference: ignored upstream source checkouts, organized as `<GitHubOwner>/<RepositoryName>`
- third_party: legacy dependency paths and version records retained for existing consumers

## Current status

As of 2026-11-06, X2 research has progressed into real MuJoCo/MC standing,
bounded locomotion, table-approach and manipulation integration. These are
experiment-specific capabilities, not a blanket qualification of navigation,
perception, manipulation or hardware deployment. Unity remains the intended
high-fidelity frontend; its planned role is not evidence of completed integration.

The active Issue #46 milestone is fixed-stance bottle grasping with the reviewed
`X2-OmniPicker-STL` lower concave-pad support circle. Standing/docking and static
IK have passing local evidence; actual jaw opening and pose tracking still block
a qualified bilateral grasp/lift. No >=50 mm lift plus >=1 s unsupported hold
has passed in this current setup.

Read the [X2 task workflow](docs/workflows/06_X2_AUTONOMY_TASK.md) for task-stage
scope and the [measured grasp findings](docs/research/issue46_truth_grasp_execution_findings.md)
for exact evidence and limitations. The [project plan](docs/00_PROJECT_PLAN.md)
P0–P7 labels are platform workstreams, not the user's X2 task Stage 2/3 labels.
