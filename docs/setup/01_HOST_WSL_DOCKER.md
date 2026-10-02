# Host Setup: Windows + WSL2 + Docker Desktop

## Host responsibility

The WSL host should stay thin. Do not install a second Docker Engine inside Ubuntu.

The host provides:
- WSL2 Ubuntu 22.04
- Docker CLI integration
- Git / editor
- Windows NVIDIA driver

The container provides:
- ROS 2 Humble
- build toolchain
- MuJoCo
- Unitree SDK / ROS packages
- project dependencies

## Project location

Use the WSL Linux filesystem, not `/mnt/c`:

```bash
~/projects/RobotSim
```

This avoids NTFS bind-mount performance and Linux permission issues.

## Quick checks

```bash
docker version
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```
