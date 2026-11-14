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

## X2 runtime clock and mode preflight

Before an expensive Issue #46 native runtime episode, run the read-only
infrastructure check from the assigned WSL task checkout:

```bash
python3 -B -m scripts.research.issue46.native_runtime.infrastructure_preflight
```

The native reset/stand entrypoint also runs this check before building the
observer or starting a simulator/MC container. It checks PTY availability and
observes `CLOCK_MONOTONIC_RAW`, `CLOCK_MONOTONIC`, and `CLOCK_REALTIME` for
35 seconds. A failed check records `physical_episode_started: false` and leaves
SIMULATOR/MC as NOT_RUN. It never repairs mounts, changes host time, or restarts
WSL/Docker. Passing this short window does not establish long-run stability;
existing runtime freshness and clock-fault stopping gates still apply.

The 100 ppm rate difference and 10 ms wall-clock step bounds are conservative
**infrastructure diagnostics**, not vendor joint/motion limits or a substitute
for synchronization qualification. The November 14 affected host measured a
3.13% MONOTONIC/RAW rate difference and wall-clock backsteps up to 558 ms.
Do not disable the check or relax physical safety gates to continue a rollout.
Do not apply repeated `date`, `adjtimex`, time-service changes, or WSL restarts
as automatic recovery. Preserve the report and investigate synchronization
ownership across the shared WSL2 VM in a separately bounded maintenance window.

Upstream status checked 2026-11-14:

- [Docker Desktop #761](https://github.com/docker/desktop-feedback/issues/761)
  describes shared-WSL clock steps; a collaborator reported a fix in progress
  on November 2. This is evidence of a similar upstream issue, not proof of the
  local setter's ownership or a released patch for Docker Desktop 4.93.0.
- [rmw_fastrtps #842](https://github.com/ros2/rmw_fastrtps/issues/842) describes
  service discovery/response races on Humble/Jazzy. A missing SetMcAction reply
  does not establish that the action was unapplied, nor identify this race as
  the cause of a particular local MC timeout.

The research clients embed `native_runtime/mode_handshake.py`: reuse persistent
clients, retain one executor owner, issue at most one state-changing request,
and verify a fresh GetMcAction response. Delivered `McActionStatus` values are
IDLE=0, RUNNING=100, TRANSITION=200; a matching name in TRANSITION is not ready.
An already RUNNING requested mode needs no repeated Set. A missing response may
be resolved only by a valid RUNNING readback; explicit rejection still fails.
Record acknowledgement and applied mode separately. Applied mode still does
not prove physical posture, selected velocity input, or safe motion.

Disposition: these two helpers remain research-only tools for the existing
native-runtime entrypoint; production code must not import them. No robot
model, actuator, collision, velocity, or joint-limit policy is changed. The
isolated generated-message regression uses no MC/simulator or velocity
publisher and is not a physical manipulation qualification.
