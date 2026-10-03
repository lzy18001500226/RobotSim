# Workflow 04: G1 SDK2 LowCmd/LowState Round Trip

## Scope and result

This workflow records the local Issue #9 round-trip experiment and the
standalone-backend evidence collected for Issue #10. It does not select a
production topology or transport, change ADR-0005, validate Unity, or claim
real-time behavior. No real hardware was used.

The pinned standalone Unitree G1 simulator accepted bounded SDK2 `LowCmd`
messages and returned CRC-valid `LowState` messages whose `tau_est` followed
the commanded torque sign. The 60-second run passed all 120 phase observations.
Source inspection also found unsynchronized concurrent access to shared
MuJoCo `mjData`; this is a serious upstream integration risk and was not
patched in the vendor checkout.

## Locked setup

| Component | Tested value |
| --- | --- |
| RobotSim base | `88b85c4446de08a2c225ab1dabf8830e652e991d` |
| `unitree_sdk2` | `63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36` |
| `unitree_mujoco` | `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` |
| `unitree_ros2` | `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` (pinned, not used) |
| Native MuJoCo | `3.3.6`, from `robotsim:humble-dev` |
| Development image | `robotsim:humble-dev`, digest `sha256:5dae0a80a15aa111b122ae3d0ad5a4022384f7b0bfcb84ddc6dd406e0f37d1f9` |
| Host runtime | WSL2 Ubuntu 22.04.5, kernel `6.6.87.2-microsoft-standard-WSL2` |
| DDS isolation | Domain `73`, interface `lo`, simulator and probe in the same Docker container |

The pinned vendor source was mounted read-only at `/vendor`. The simulator,
SDK install, and probe build were kept in `/lab` or a derived build overlay;
the vendor checkouts were not modified. `unitree_ros2` is pinned in
`third_party/LOCK.md` but is not part of this direct SDK2 DDS test.

## Reproduction

Fetch the repository-pinned source into a scratch location, not the tracked
source tree:

```bash
scripts/fetch_third_party.sh /tmp/robotsim-issue9-sources/third_party
```

In the Humble development container, build the SDK to an external prefix, then
build the pinned simulator from a writable overlay so its MuJoCo link/config
does not alter the read-only upstream checkout. Build the RobotSim-owned probe
against that SDK prefix:

```bash
cmake -S /vendor/unitree_sdk2 -B /lab/sdk-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/lab/sdk-prefix \
  -DBUILD_EXAMPLES=OFF
cmake --build /lab/sdk-build -j2
cmake --install /lab/sdk-build

mkdir -p /lab/sim-overlay
cp -a /vendor/unitree_mujoco /lab/sim-overlay/unitree_mujoco
ln -sfn /opt/mujoco/mujoco-3.3.6 \
  /lab/sim-overlay/unitree_mujoco/simulate/mujoco
cmake -S /lab/sim-overlay/unitree_mujoco/simulate \
  -B /lab/sim-overlay/unitree_mujoco/simulate/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /lab/sim-overlay/unitree_mujoco/simulate/build -j2

cmake -S robots/unitree_g1/tools/lowcmd_roundtrip \
  -B /lab/probe-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/lab/sdk-prefix
cmake --build /lab/probe-build -j2
```

Start the pinned 29-DoF G1 scene from the simulator build directory, then run
the bounded probe. The maintained wrapper
[`scripts/dev/run_g1_lowcmd_roundtrip.sh`](../../scripts/dev/run_g1_lowcmd_roundtrip.sh)
sets the isolated domain/interface and forwards the remaining probe arguments.

```bash
cd /lab/sim-overlay/unitree_mujoco/simulate/build
./unitree_mujoco -r g1 -i 73 -n lo -s scene_29dof.xml -t 1
/lab/probe-build/lowcmd_roundtrip_probe \
  --domain 73 --interface lo --seconds 60 --publish-hz 100 \
  --phase-ms 500 --joint 18 --tau-nm 0.10 \
  --output-prefix /lab/roundtrip-60s
```

The probe publishes to `rt/lowcmd`, subscribes to `rt/lowstate` with SDK2
queue length 1, checks the LowState CRC, and records monotonic and wall-clock
timestamps. It sets zero gains and zero torque on all motors except joint 18,
which alternates between `+0.10 Nm` and `-0.10 Nm` every 500 ms. The positive
torque bound is capped at `0.20 Nm`; duration is capped at 120 s and publish
rate at 250 Hz. A phase counts as observed only when a later CRC-valid state
reports matching `tau_est` magnitude of at least `0.05 Nm` before the next
phase. This is transport and actuator-feedback evidence only; it is not a
whole-body stability, walking, or control-quality test.

## Measured standalone run

The corrected 60-second run started at `2026-10-03T09:14:24.520Z` and ended
with `PASS`. The CSV records use `steady_clock` monotonic timestamps; wall-clock
timestamps are retained for correlation. Percentiles use nearest-rank
selection. Command-to-state latency is measured from immediately before
`publisher.Write()` to the first matching `LowState.tau_est` callback in each
500 ms phase.

| Measurement | Result |
| --- | ---: |
| Simulator update ticks | `999.995 Hz` |
| Received `LowState` callback rate | `689.116 Hz` (`41,486` samples) |
| Published `LowCmd` rate | `99.999 Hz` (`6,000` samples, zero write failures) |
| Valid LowState CRC / capture drops | `41,486 / 41,486`; `0` drops |
| Observed torque phases | `120 / 120`; `0` missed |
| LowCmd-to-observed-`tau_est` latency p50/p95/p99 | `3.500 / 7.594 / 11.877 ms` |
| LowState interval p50/p95/p99 | `1.272 / 2.498 / 9.092 ms` |
| Absolute LowState interval error from 1 ms p50/p95/p99; max | `0.292 / 1.498 / 8.092 ms`; `17.460 ms` |
| LowCmd interval p50/p95/p99 | `10.006 / 13.995 / 16.965 ms` |
| Absolute LowCmd interval error from 10 ms p50/p95/p99; max | `0.251 / 5.341 / 8.618 ms`; `15.127 ms` |
| Late LowState intervals (`>1.5 ms`) | `3,476` |
| Late LowCmd intervals (`>15 ms`) / late publish calls | `159 / 25` |
| Repeated simulator ticks / skipped tick count | `16,402 / 35,117` |
| Container CPU samples, mean / min..max | `649.62% / 615.47..662.10%` (7 samples, 10 s apart) |
| Container memory | `1.001..1.034 GiB` |

The p99 latency is an empirical tail percentile over 120 transitions, not a
confidence bound. The interface emitted a warning that `lo` is not
multicast-capable and DDS multicast was disabled; LowCmd/LowState traffic still
worked inside the isolated container. The received state rate is below the
simulator tick rate, with repeated and skipped tick values. The SDK2 subscriber
queue is depth 1 and the probe recorded no capture overflow or failed writes;
the upstream DDS implementation exposes no queue/backpressure counters here,
so middleware backlog and loss are not fully observable from this test.

Docker exposed an RTX 4060 Laptop GPU to the development container, but
`glxinfo -B` reported Mesa `llvmpipe`. The simulator GUI was therefore using
software rendering. CPU figures are whole-container readings under WSLg and
must not be interpreted as isolated physics cost or GPU-accelerated
performance. No deterministic, bounded-latency, or real-time guarantee follows
from these measurements.

## Lifecycle and failure isolation

- Initial startup logged MuJoCo `3.3.6`, `Mujoco data is prepared`, and the
  `unitree_hg` IDL selection for `g1`.
- In the failure-injection run, the simulator was terminated while a bounded
  probe was active. After `LowState` age reached `757 ms`, the probe reported
  `FAIL: LowState timeout` and ended the round trip as `FAIL` rather than
  continuing to publish indefinitely.
- Restarting the same pinned simulator in domain 73 restored the channel. The
  final 3-second restart probe passed with valid CRCs and `6/6` observed torque
  phases.
- SIGTERM shutdown completed; exact-name process checks found no
  `unitree_mujoco` or `lowcmd_roundtrip_probe` process afterward. The temporary
  lab container is removed during task cleanup. No real DDS participant or
  physical robot was involved.

## Shared `mjData` concurrency finding

At pinned `unitree_mujoco` commit
`1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`:

- [`RobotBridge::start/run`](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/simulate/src/unitree_sdk2_bridge.h#L168)
  runs a 1 kHz recurrent thread. Its `lowcmd->mutex_` protects the DDS command
  message only; it reads/writes `mj_data_->ctrl` and reads
  `mj_data_->sensordata` and `mj_data_->time` without taking the simulator's
  mutex.
- [`PhysicsLoop`](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/simulate/src/main.cc#L411)
  holds `sim.mtx` while advancing the same `mjData` with `mj_step()`.
  Because the bridge thread does not acquire that lock, normal execution has
  unsynchronized concurrent access to shared `mjData` (`ctrl` and sensor/time
  fields): a source-level C++ data-race risk, not a race dynamically confirmed
  by ThreadSanitizer.
- [`main()` and the bridge startup loop`](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/simulate/src/main.cc#L573)
  start the bridge thread before the physics thread; the bridge polls global
  `d` while the physics thread initializes it, without synchronized
  publication. The bridge then retains raw model/data pointers and has no
  stop/join path; the physics thread exits the process when its loop ends.

No vendor source was changed and no synchronization workaround was added.
The probe cannot safely acquire the simulator's private `sim.mtx`; resolving
this requires a RobotSim-owned integration/lifecycle boundary that shares an
explicit data-access lock. Treat the current bridge as unsuitable for a
production concurrency guarantee until that work is separately designed and
validated.

## Validation and remaining boundary

- `python3 -m unittest discover -s tests -v`: 12 repository tests passed.
- Python compilation and `bash -n` for the touched scripts: passed.
- SDK2 and RobotSim probe CMake builds in `robotsim:humble-dev`: passed.
- `scripts/check_dev_container.sh`, including required NVIDIA visibility:
  passed; software-rendering limitation noted above.
- `scripts/run_g1_mujoco_smoke.sh` in an isolated Python `3.12.15` container:
  4 tests passed; both pinned G1 variants loaded, stepped, and responded to
  bounded input with MuJoCo `3.3.6`.
- The direct SDK2/DDS 60-second round trip and restart/failure tests above:
  passed for their stated conditions.
- `git diff --check`: run again at commit closeout.

Raw CSV and lifecycle logs are retained outside the checkout in
`~/.local/share/robotsim/issue9-standalone-evidence/2026-10-03/`, with a
`SHA256SUMS` manifest. They are intentionally not committed as benchmark
traces. Issue #9's local simulated round trip is evidenced; standalone
evidence for Issue #10 is collected, but the Unity-hosted comparison and ADR
decision remain deferred. ADR-0005 is unchanged. No Issue #11 implementation
was started.
