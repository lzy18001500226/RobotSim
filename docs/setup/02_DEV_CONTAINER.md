# Development Container

## Why Docker is the source of truth

The project is intended to be reproducible on:
- WSL2 development machines
- native Ubuntu 22.04 workstations
- future lab servers

Do not rely on a WSL export as the primary reproducibility mechanism.

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
