# Workflow 01: G1 Baseline

## Goal

Bring up the first Unitree G1 backend as a minimal RobotSim end-to-end baseline.

This workflow validates the platform architecture before adding high-fidelity sensors, navigation, or manipulation.

## Preconditions

- WSL2 Ubuntu 22.04 is working.
- Docker is working from WSL.
- NVIDIA GPU access is verified where required.
- GitHub access is configured.
- ROS 2 Humble baseline is available.
- A MuJoCo release has been selected and pinned for the current integration.
- Unitree upstream dependencies are recorded in `third_party/LOCK.md`.

## Steps

1. Fetch the pinned Unitree G1 dependencies.
2. Build the required Unitree SDK and simulator components.
3. Verify that the selected MuJoCo release loads correctly.
4. Load the official G1 model without modifying upstream vendor files in place.
5. Start the MuJoCo simulation.
6. Verify simulation stepping and robot stability.
7. Verify joint state and actuator access.
8. Verify Unitree DDS / SDK2 communication required by the selected backend.
9. Expose the minimum RobotSim ROS 2 state and command interfaces.
10. Verify that simulation ground truth and estimated state are separate.
11. Record the exact dependency commits and MuJoCo version.
12. Only after the baseline is stable, add higher-level capabilities.

## Minimum Validation

The baseline is considered working when:

- the G1 model loads without fatal errors
- MuJoCo physics advances continuously
- joint states can be read
- actuator commands can be sent
- the robot remains physically stable in the intended initial mode
- the ROS 2 baseline can observe robot state
- robot-specific SDK usage remains inside the G1 backend
- the procedure can be repeated from a clean environment

## Not Part of This Baseline

Do not add these until the minimum baseline is stable:

- grasping
- LiDAR
- SLAM
- navigation
- RGB-D perception
- Unity high-fidelity rendering
- VLA or reinforcement-learning training

These capabilities are later validation layers, not prerequisites for proving the G1 physics and interface baseline.
