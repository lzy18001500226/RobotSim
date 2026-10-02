# ADR-0003: G1 first, X2 second

Status: Accepted

## Decision

Use Unitree G1 as the first complete RobotSim robot backend, then add AgiBot X2.

## Rationale

G1 currently has a mature public MuJoCo, ROS 2, perception, navigation, and manipulation ecosystem. This makes it suitable for validating the RobotSim architecture before introducing a second robot backend.

AgiBot X2 remains an important target because its software stack, model structure, sensor configuration, and runtime interfaces differ from G1.

Supporting both robots will expose which interfaces can genuinely be shared and which must remain robot-specific.

The architecture must remain robot-independent above the adapter layer.

## Consequences

- Initial integration work may contain G1-specific implementation inside `robots/unitree_g1/`.
- G1-specific assumptions must not leak into task, perception, localization, navigation, or generic manipulation packages.
- Shared interfaces should be generalized after the X2 backend exposes real differences, not before.
