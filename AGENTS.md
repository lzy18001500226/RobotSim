# Project Agent Instructions

## Core rule
Do not put vendor-specific logic in task, perception, localization, navigation, or generic manipulation packages.

Vendor SDK usage belongs only in robot adapters:
- `robots/unitree_g1/`
- `robots/agibot_x2/`

## Source of truth
- Environment: Dockerfiles + Compose
- ROS dependencies: rosdep / apt
- Third-party source versions: pinned commits in `third_party/LOCK.md`
- Design decisions: `docs/adr/`
- Repeatable procedures: `docs/workflows/`
- Technology notes: `docs/skills/`

## Development rules
- Keep simulation truth and estimated state separate.
- Every sensor simulator must publish a ROS 2 compatible interface.
- Every algorithm must be runnable with a simulated backend first.
- Do not edit upstream vendor robot files in place; use overlays/adapters.
- New robot backends must preserve the common interfaces.
