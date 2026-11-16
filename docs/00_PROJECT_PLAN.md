# RobotSim Project Plan

## Goal

Build a reusable robotics simulation and system-integration platform for multiple robots, sensors, environments, and tasks.

RobotSim should support the same high-level software stack across simulation and real robot deployment whenever possible.

Initial robot platforms:

- Unitree G1
- AgiBot X2

Initial application scenarios include indoor navigation, perception, manipulation, and object fetching, but these tasks validate the platform rather than define its architecture.

## Core Architecture

RobotSim is divided into three main layers.

### MuJoCo

MuJoCo is the physics source of truth and is responsible for:

- articulated robot dynamics
- joints and actuators
- contacts and collision response
- physical state
- proprioceptive and internal robot state
- simulation ground truth

### Unity

Unity is the high-fidelity frontend and is responsible for:

- high-fidelity environments
- visual rendering
- RGB and depth cameras
- external sensor simulation
- visualization and scene interaction

Unity must not become an independent second physics source for the robot.

### ROS 2

ROS 2 is the system-integration layer and is responsible for:

- sensor and robot interfaces
- perception
- localization and mapping
- navigation
- manipulation
- task orchestration
- simulation and real-robot integration

## Robot Backends

Robot-specific logic belongs in adapters.

Initial backends:

- Unitree G1
- AgiBot X2

Higher-level algorithms should not depend directly on vendor SDKs.

Do not design a large universal robot abstraction before at least two real backends have been exercised. Start with G1, add X2, then generalize only the interfaces that are genuinely shared.

## Progress terminology (updated 2026-11-06)

The P0–P7 sections below are the original platform roadmap, not a current
sequential task queue or completion checklist. X2 native MuJoCo/MC work is
already underway; do not defer it until every earlier roadmap item is finished.
No generic roadmap section is declared PASS by this update.

For the current user-defined X2 task stages:

- **Stage 2:** preplan a feasible fixed table stance; travel there; execute
  open, approach, measured arrival, close, lift and unsupported hold using the
  reviewed lower concave-pad circle. Prefer fresh depth-camera targets; the
  maintainer permits truth-first grasp debugging followed by depth-camera
  replacement. Truth-based success must remain separately labeled.
- **Stage 3:** know the table but not bottle position in advance; use fresh
  depth perception to choose/adjust the base stance fore/aft/laterally and
  repeat grasp planning. A fixed docking coordinate is not this capability.
- LiDAR localization and unknown-obstacle navigation are distinct qualification
  targets; passing ground-truth routes does not prove them.

Stage 2 is currently incomplete: the measured opening and pose-tracking blockers
are recorded in [Issue #46 findings](research/issue46_truth_grasp_execution_findings.md).
A geometry review or static IK PASS does not establish grasp/lift success.

## Development Phases

### P0 - Development environment

Establish and verify:

- Windows 11 host
- WSL2 Ubuntu 22.04
- Docker
- ROS 2 Humble
- NVIDIA GPU access from WSL and Docker
- Git and GitHub SSH
- Unity Linux Editor experiment under WSLg

### P1 - MuJoCo baseline

Validate:

- pinned MuJoCo release
- basic robot model loading
- physics stepping
- actuator commands
- joint and state access
- simulation clock
- ROS 2 interface baseline

Use Unitree G1 as the first robot backend.

### P2 - Unity frontend

Validate:

- Unity Editor in the selected development environment
- MuJoCo Unity integration
- synchronized robot visualization
- scene loading
- RGB rendering
- depth rendering
- external sensor frontend architecture

### P3 - Robot-independent ROS 2 interfaces

Define and exercise ROS 2 interfaces for:

- robot state
- control commands
- sensor data
- simulation clock
- ground truth
- estimated state

Simulation truth and estimated state must remain separate.

### P4 - Perception and localization

Add:

- RGB-D perception
- LiDAR simulation
- mapping
- localization
- sensor timing and frame conventions

### P5 - Navigation and manipulation

Add reusable capabilities such as:

- navigation
- reaching
- grasping
- pick and place
- object fetching

These tasks validate the platform but do not define its architecture.

### P6 - AgiBot X2 backend

Integrate AgiBot X2 while preserving the same high-level ROS 2 interfaces where practical.

Differences between G1 and X2 should remain in robot-specific adapters and configuration.

### P7 - Sim2Real

Replace simulated backends with real robot and sensor interfaces while preserving higher-level algorithms wherever possible.

The target is not zero difference between simulation and reality. The target is a stable software boundary that allows simulated and real backends to be exchanged with minimal changes above the adapter layer.

## Design Principles

- MuJoCo is the physics source of truth.
- Unity is the high-fidelity scene and external sensor frontend.
- ROS 2 is the system-integration layer.
- Simulation truth and estimated state must remain separate.
- Vendor SDK code must stay inside robot-specific adapters.
- Avoid maintaining duplicate physics worlds.
- Keep one authoritative physical scene representation.
- Prefer reusable interfaces over task-specific coupling.
- Prefer upstream vendor and open-source components over unnecessary reimplementation.
- Keep high-frequency low-level control close to the physics backend when practical.
- Do not over-generalize interfaces before multiple robot backends have been exercised.

## Initial Non-goals

- large-scale VLA training
- building a new physics engine
- replacing ROS 2 with custom middleware
- reproducing every real sensor imperfection in the first milestone
- reimplementing vendor functionality that can be reused safely
