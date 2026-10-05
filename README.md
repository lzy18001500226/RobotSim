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
- third_party: external dependencies and version records

## Current status

RobotSim is currently in early platform bring-up.

The current focus is validating the development environment and the MuJoCo, Unity, and ROS 2 integration architecture before implementing higher-level robot tasks.

## Issue #43 M0 Bottle Pick-and-Place

On WSL2 Ubuntu 22.04, install `git`, `uv`, and an EGL-capable Mesa runtime, then run this exact command from a fresh checkout:

```bash
./scripts/run_m0_pick_place.sh
```

The first run needs public network access for fetching the pinned Humanoid VLA and Unitree MuJoCo checkouts and installing the pinned Python packages. The launcher creates a Python 3.10 environment under `/tmp/robotsim-issue43-m0/venv`, verifies existing environments are Python 3.10.x, and refuses dirty or wrong-commit upstream checkouts. `ROBOTSIM_M0_MESH_DIR`, if set, must resolve inside the verified pinned Unitree `unitree_robots/g1/meshes` tree; external mesh paths are rejected.

Each invocation gets a unique run ID and output directory beneath `/tmp/robotsim-issue43-m0/output/`. It creates `m0_result.json` as `PREFLIGHT` with `passed: false` before checks begin, and leaves a per-invocation failure result on preflight or runtime errors. A completed run writes `m0_result.json`, `m0_physics_trace.jsonl`, `m0_pick_place.mp4`, `m0_final.png`, and `run.log`; generated scene files are kept under that run's `model/` directory. Set `ROBOTSIM_M0_RUN_DIR` to select the checkout/cache/venv root, `ROBOTSIM_M0_OUTPUT_DIR` to select the output root, `ROBOTSIM_M0_CANDIDATE_DIR` or `ROBOTSIM_M0_UNITREE_DIR` to select existing pinned checkouts, and `ROBOTSIM_M0_SEED` to select the seed.

The generated model restores the paired stock left and right rubber hands directly from the pinned Unitree G1 model, without scale or wrist-transform overrides. The previous Dex3 overlay caused the abnormal appearance. The stock meshes have no finger joints, hand actuators, or hand collision geometry, so they cannot satisfy the physical grasp requirement. The launcher therefore renders a morphology/scene review and records `BLOCKED`; it does not run or claim a grasp or lift.

The review scene preserves the pinned G1 camera, lighting, skybox, and dark checker floor, and uses the existing G1 table footprint. The bottle is copied exactly from the X2 grasp script: one free-jointed cylinder, 75 mm diameter, 240 mm tall, 0.57 kg, blue RGBA `[0.12, 0.52, 0.82, 1.0]`, friction `[1.4, 0.02, 0.001]`, and `condim=4`. It rests directly on the tabletop with no pedestal, weld, or runtime qpos write.

Each run writes `m0_result.json`, `m0_physics_trace.jsonl`, `g1_stock_hand_scene_review.mp4`, `g1_stock_hand_overview.png`, `run.log`, and overview, paired-hand, bottle, and right-hand near-bottle reference images under `screenshots/`. The reference view records right-arm IK status and measured hand-to-bottle distance; it is not a completed pre-grasp unless the result says the approach pose was planned. The bottle remains under free physics, and no lift is attempted because the stock hands lack articulation and collision geometry. Physics runs on CPU with offscreen EGL rendering and Mesa; no GPU, display, ROS 2, Unity, network transport, or physical hardware is required after setup.
