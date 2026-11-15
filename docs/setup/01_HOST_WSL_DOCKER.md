# Host Setup: Windows + WSL2 + Docker Desktop

## Host responsibility

The WSL host should stay thin. Do not install a second Docker Engine inside Ubuntu.

The host provides:
- WSL2 Ubuntu 22.04
- Docker CLI integration
- Git / editor
- Windows NVIDIA driver

The selected container profile provides:
- ROS 2 Humble
- build toolchain
- MuJoCo
- Unitree SDK / ROS packages
- project dependencies

The X2 review workflow also uses the existing host virtual environment
`/home/lzy18001500226/.venvs/robotsim-mujoco336` for its launcher, model
inspection and viewer tools. This is a scoped workflow dependency, not a reason
to install ROS or another Docker daemon into the host. Use the interpreter
specified by the runner; do not repair imports through global Python or an
ambient `PYTHONPATH`. See [runtime profiles](02_DEV_CONTAINER.md#choose-the-runtime-profile).

## Project location

Use the WSL Linux filesystem, not `/mnt/c`:

```bash
~/projects/RobotSim
```

This avoids NTFS bind-mount performance and Linux permission issues.

## Read-only preflight

Run from the assigned WSL worktree:

```bash
docker version
docker context show
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

Confirm a reachable Linux server, the runner's pinned image and no conflicting
experiment. A working Windows Docker CLI does not prove that Ubuntu's CLI
integration works. If the WSL CLI or socket is unavailable, inspect Docker
Desktop's Linux/WSL engine and **Resources -> WSL Integration -> Ubuntu-22.04**.
Do not install a second daemon or restart WSL/Desktop underneath another task.
Recheck from the same WSL distribution used by the runner.

The following start disposable containers. Use them for initial setup or a
specific engine/GPU diagnosis, not as mandatory checks on every conversation:

```bash
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

## Container lifecycle and cleanup

An image is a reusable dependency snapshot. A container is one execution plus
its writable layer; stopping it does not remove that layer. The X2 workflow
normally needs one selected runtime image and one temporary episode container,
not a permanent container per Issue, chat or failed run. Other robots and GPU
tools may have separate images; image count is not experiment count.

- One owner controls the live X2 episode. Give it a unique attempt/container
  name and bind-mount source, configuration and governed evidence explicitly.
- Keep source/build outputs and logs in their intended host paths. Do not rely
  on a stopped container as the only copy of code, SDK installs or evidence.
- At episode exit, collect evidence and remove only that owned container.
  Verify absence through a successful Docker listing or another explicit
  absence check; an arbitrary failed `docker inspect` is not proof of removal.
- Before removing historical containers, inspect their identity, state, mounts,
  consumers and `docker diff`. Save unique logs/configuration/builds and a
  recovery manifest outside the container. Preserve host bind sources.
- Remove reviewed stopped containers by exact ID with `docker rm <id>`.
  Do not force-remove running experiments or use `-v`, broad `prune`, image
  deletion or volume deletion as routine cleanup. Those require a separate
  dependency/data review.

Docker references: [WSL integration](https://docs.docker.com/desktop/features/wsl/)
and [resource pruning semantics](https://docs.docker.com/engine/manage-resources/pruning/).
