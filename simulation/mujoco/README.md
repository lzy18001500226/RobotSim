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
