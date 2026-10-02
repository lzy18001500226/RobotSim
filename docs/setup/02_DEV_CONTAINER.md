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

## Third-party build persistence

Vendor source and compiled outputs live under the bind-mounted project tree.
`unitree_sdk2` is installed to:

```text
third_party/install/unitree_robotics
```

so it survives disposable `docker compose run --rm ...` containers.

## Important

The first image is intentionally minimal. Heavy model stacks such as FoundationPose should later use separate GPU containers.
