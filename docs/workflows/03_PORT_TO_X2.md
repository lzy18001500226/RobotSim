# Workflow: Port G1 Stack to X2

## Keep unchanged
- task manager
- navigation API
- perception API
- common object representation
- evaluation tooling

## Replace
- robot MJCF / URDF
- robot adapter
- actuator / hand interface
- sensor profile
- vendor runtime

## X2-specific runtime
The expected runtime boundary is:
AimDK -> MC -> sim_mujoco (simulation)
AimDK -> MC / HAL -> physical X2 (real robot)

Do not edit upstream X2 model files in place. Use overlays and configuration.
