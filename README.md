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

On WSL2 Ubuntu 22.04 with `git`, `uv`, public network access for the first checkout/package install, and a system EGL/Mesa runtime for offscreen rendering, run `./scripts/run_m0_pick_place.sh`. No GPU, visible display, ROS 2, or Unity runtime is required. The launcher verifies pinned Humanoid VLA and Unitree MuJoCo commits, enforces Python 3.10, and creates the runtime under `/tmp/robotsim-issue43-m0`. The adapter writes the generated scene outside the repository and does not modify either upstream checkout.

The launcher writes `m0_result.json`, `m0_physics_trace.jsonl`, `m0_pick_place.mp4`, `m0_final.png`, and `run.log` under `/tmp/robotsim-issue43-m0/output`. The bottle uses stacked cylindrical collision geoms. Acceptance requires bilateral palm force contact, a 5 cm lift, every bottle collision geom inside the target tabletop with 3 cm edge clearance, contact-free release, target-table contact, and one second of stable target contact. Upright orientation is reported but is not required. It also checks finite/bounded state, penetration at each physics step, drop, per-step pose discontinuities, and weld-event pose jumps. The JSON includes the RobotSim and upstream revisions, geometry, thresholds, weld-event poses/times, and whole-object footprint evidence. The run is CPU MuJoCo physics with offscreen EGL rendering; it requires no ROS 2, Unity, network transport, or physical hardware.

Set `ROBOTSIM_M0_RUN_DIR`, `ROBOTSIM_M0_OUTPUT_DIR`, `ROBOTSIM_M0_CANDIDATE_DIR`, or `ROBOTSIM_M0_UNITREE_DIR` to select external cache/output locations. `ROBOTSIM_M0_MESH_DIR` is accepted only when it resolves to the mesh directory inside the verified pinned Unitree checkout; arbitrary external meshes are rejected. Existing checkouts must already be at the exact pinned commit and clean; the launcher refuses to switch or overwrite a mismatched checkout. Every invocation overwrites the result with a new non-PASS preflight record before dependency/setup work, and the final JSON records the run ID, RobotSim/upstream identities, Python/package versions, mesh provenance, seed, thresholds, and output directory.
