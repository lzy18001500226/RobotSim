# Development Container

## Why Docker is the source of truth

The project is intended to be reproducible on:
- WSL2 development machines
- native Ubuntu 22.04 workstations
- future lab servers

Do not rely on a WSL export as the primary reproducibility mechanism.

## Choose the runtime profile

Use the task's existing launcher and pinned environment. The generic development
container below is not a replacement for the validated X2 runtime.

| Profile | Purpose and ownership |
| --- | --- |
| X2 RAICOM | The active Issue #46/#80 integration uses `robotsim:humble-raicom-runtime-20261019`. Its `stl_hand_startup.py` checks image ID prefix `fed10384dea8`, launches an isolated episode and owns cleanup. Keep its ROS domain 20 and Fast DDS settings with that runner. |
| Generic Humble development | `docker/compose.yaml` builds `robotsim:humble-dev` for the shared toolchain and earlier G1 work. Use only when that task needs it; its networking/RMW settings do not define the X2 experiment contract. |
| Historical/optional tools | Grove/Jazzy, FoundationPose/PyTorch and GPU diagnostic images are separate task dependencies. Retain or remove them after inspecting consumers; do not launch them as X2 startup prerequisites. |

The X2 launcher is currently on the assigned integration branch under
`scripts/research/issue46/native_runtime/stl_hand_startup.py`; it is not
present in every infra checkout. Resolve the actual worktree with Git and read
its launcher rather than guessing a path or copying its invocation here.

At the 2026-11-15 inspection, the local RAICOM image resolved to
`sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31`.
This is a local image identity, not a registry pull digest or permission to
rebuild/replace it. A mutable tag alone is insufficient provenance. Preserve
the vendor delivery, model/configuration hashes and evidence paths alongside
the image identity; do not commit proprietary packages into Git.

Use [host preflight and cleanup](01_HOST_WSL_DOCKER.md#container-lifecycle-and-cleanup)
for container retention. An empty container list between episodes is normal.
The generic commands below build/start containers and are opt-in setup steps,
not commands that every agent should run before an unrelated task.

## Build

From the project root:

```bash
docker compose -f docker/compose.yaml build
```

## Enter

```bash
docker compose -f docker/compose.yaml run --rm dev
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
ros2 --help
mujoco_version
```

## Smoke verification

From the project root, run the repeatable toolchain and native-library checks:

```bash
docker compose -f docker/compose.yaml run --rm -T dev bash scripts/check_dev_container.sh
```

To require NVIDIA GPU visibility in a GPU-enabled container, add
`-e ROBOTSIM_REQUIRE_NVIDIA_GPU=1` after `run --rm -T`.

### Local acceptance record (2026-10-03)

On WSL2 Ubuntu 22.04 with Docker Engine 29.8.1 and Compose 5.5.1:

- `docker compose -f docker/compose.yaml build` completed.
- The `osrf/ros:humble-desktop-full` tag resolved during this build to
  `sha256:1db1e4e941d4f77fab55bcd479158a75273329335e86ec468bff280859e8178c`.
- The local `robotsim:humble-dev` image ID is
  `sha256:5dae0a80a15aa111b122ae3d0ad5a4022384f7b0bfcb84ddc6dd406e0f37d1f9`.
- The GPU-required smoke command passed: ROS 2 Humble and `rclpy`/`std_msgs`
  imports, CMake 3.22.1, Ninja 1.10.1, loaded native MuJoCo 3.3.6, and an
  NVIDIA GeForce RTX 4060 Laptop GPU were observed.

This records one successful local build, not a bit-for-bit reproducible image:
the base image is referenced by a mutable tag and apt package versions are not
pinned.

## Third-party build persistence

Vendor source and compiled outputs live under the bind-mounted project tree.
`unitree_sdk2` is installed to:

```text
third_party/install/unitree_robotics
```

so it survives disposable `docker compose run --rm ...` containers.

## Important

The first image is intentionally minimal. Heavy model stacks such as FoundationPose should later use separate GPU containers.
