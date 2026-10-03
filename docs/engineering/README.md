# RobotSim engineering contract

This page is the index for implementation and validation conventions. It applies to the current early bring-up stage; it does not assert that planned packages or runtime interfaces already exist. Read the focused [runtime contract](runtime.md), [validation contract](validation.md), and [Goal workflow](../workflows/goal_driven_development.md) when they apply.

The architecture boundary is already established in [ADR-0004](../adr/0004_mujoco_unity_ros2_architecture.md): MuJoCo owns robot physics and ground truth, Unity owns scene/rendering/external sensor simulation, and ROS 2 integrates simulated and real backends. These engineering rules make that boundary actionable; they do not choose a new physics topology.

## Where work belongs

| Path | Responsibility |
| --- | --- |
| `apps/` | User-facing applications. `apps/unity/` owns Unity scenes, rendering, sensor-front-end code, and editor tooling. |
| `robots/` | Robot descriptions and backend adapters, organized by platform such as `unitree_g1/` and `agibot_x2/`. Keep vendor SDK calls behind these adapters. |
| `ros2_ws/src/` | ROS 2 packages and shared ROS interfaces. Keep package code, `include/`, `src/`, `test/`, `launch/`, and `config/` together in each package as applicable. |
| `simulation/` | Simulation backends and their runtime integration. `simulation/mujoco/` owns MuJoCo physics/runtime responsibilities. |
| `scripts/agent/` | Repository-agent support such as hooks, notifications, and small Codex helpers. |
| `scripts/dev/` | Reusable developer setup, build, and local workflow commands. |
| `scripts/ci/` | Reusable CI checks only when workflow logic is substantial enough to share or test. Small workflow-local commands can stay in `.github/workflows/`. |
| `tests/` | Cross-repository infrastructure and tests that do not belong to one package. Runtime and ROS package tests live beside their package. |
| `tools/` | Optional maintained developer utilities with a defined owner and repeatable use. Do not add it for a one-off script. |
| `data/` | Small checked-in fixtures only when licensing and size are suitable. Keep datasets, rosbags, model weights, benchmark output, screenshots, and traces out of Git by default. |
| `third_party/` | Upstream dependencies and version records. Record tested versions/commits in `third_party/LOCK.md`; do not patch a fetched vendor checkout in place. |
| `docs/` | Architecture decisions, engineering rules, setup, and repeatable workflows. |

This map describes intended ownership; an empty or not-yet-created directory is not a reason to add placeholder structure. The current small set of scripts may stay directly under `scripts/`; introduce category directories as a group grows rather than moving files just to match the map. A runtime executable belongs in its package or runtime component even if its first experiment was a script.

## Artifact discipline

- Do not create scratch files in the repository root. Put temporary diagnostics in `/tmp` or a clearly named, explicitly ignored scratch directory.
- Promote a successful experiment to its maintained package, test, or tool location, or remove it. Do not leave variants such as `test2.py`, `demo_final.py`, `tmp.cpp`, or `fix_new.sh` behind.
- Keep ROS `build/`, `install/`, and `log/`, Python caches, Unity generated caches, editor state, rosbags, datasets, weights, benchmark output, screenshots, and profiling traces out of version control. Prefer an explicit ignore rule in the owning directory or storage outside the checkout; do not add a broad ignore that can hide source or configuration.
- Keep ROS package tests package-local. Root `tests/` is for repository-wide infrastructure checks, not a substitute for package test ownership.
- Pin external code when adopted and record the source revision in the version source of truth. Use an adapter or overlay for local integration changes. Do not add submodules or a dependency manager without a concrete need.

## Authority of rules and sources

ROS REP documents are authoritative sources for their stated conventions, but informational REPs are not automatically enforced requirements. State which conventions RobotSim adopts. ROS 2, ros2_control, MoveIt, and Nav2 documents are upstream implementation guidance, not blanket requirements for RobotSim. This repository's explicit conventions are marked as RobotSim decisions. Anthropic material is comparative inspiration only. None of these references grants permission to change an accepted ADR or to claim a capability that has not been validated.
