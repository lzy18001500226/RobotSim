# Project Plan

## Goal

Build a reusable humanoid mobile-manipulation stack that can complete:

> "Go to the target area and fetch a bottle of water."

The project starts with Unitree G1 because its MuJoCo / D435i / MID360 ecosystem is mature, then ports to AgiBot X2.

## Architecture

```text
Task Manager
    |
    +---- Navigation <---- Localization <---- LiDAR + IMU
    |
    +---- Manipulation <--- Object Pose <---- RGB-D Perception
                |
                v
          Robot Adapter
          /           \
    Unitree G1       AgiBot X2
     /     \           /     \
 MuJoCo   Real     MuJoCo   Real
```

## Phases

### P0 - Environment
Docker + Ubuntu 22.04 + ROS 2 Humble + MuJoCo + compiler toolchain.

### P1 - G1 simulator baseline
Load official G1 MJCF in `unitree_mujoco`; verify DDS / LowState / LowCmd.

### P2 - Fixed-base grasp baseline
Robot starts near the table. Bottle pose may initially come from simulation truth.
Implement reach -> pre-grasp -> grasp -> lift.

### P3 - RGB-D perception
Replace bottle truth with RGB-D detection / depth-based 3D localization.
FoundationPose is optional, only if full 6DoF object pose is required.

### P4 - Mobile manipulation
Add virtual MID360-like LiDAR, mapping/localization, and navigation.

### P5 - Natural-language task layer
Language -> structured task -> navigation -> perception -> grasp.

### P6 - X2 backend
Add X2 MJCF / AimDK / MC and E1R + Gemini335 sensor profiles while keeping higher layers unchanged.

## Non-goals for the first milestone

- No VLA training.
- No refrigerator opening.
- No photorealistic Unity / UE scene.
- No dependence on proprietary SLAM.
- No requirement to reproduce real sensor noise in P1/P2.
