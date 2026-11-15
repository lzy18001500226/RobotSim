# Windows / WSL2 / Docker ROS 2 diagnostics

Use this reference only for a failure crossing these boundaries. Reuse existing traces before adding instrumentation. The [runtime contract](../../../../docs/engineering/runtime.md) owns adopted timing rules; the [simulation-state proposal](../../../../docs/engineering/simulation_state_contract.md) is design guidance, not proof that an installed vendor binary implements its envelope.

## 1. Locate the actual runtime

Record each endpoint's process, Windows/WSL distribution/container location, ROS distribution, RMW implementation, domain, startup command and installed release. Identify the Docker context and daemon host separately from the shell running the Docker CLI. An empty WSL `/opt/ros` says nothing about ROS installed in a container.

Use only the needed read-only inventory commands, from the indicated environment:

```powershell
# Windows PowerShell
wsl.exe --version
wsl.exe -l -v
```

```bash
# Shell connected to the relevant Docker daemon; substitute the existing container name.
docker context show
docker info --format '{{.OperatingSystem}}'
docker inspect --format '{{.HostConfig.NetworkMode}} {{.HostConfig.IpcMode}} {{.HostConfig.ShmSize}}' <container>
```

Do not dump complete process environments or container configuration containing credentials. Record image identity, configured CPU/memory limits and the active DDS configuration only when relevant. Inspect an existing container; do not start MC, a simulator or a privileged diagnostic container merely to inventory it.

Draw the actual boundary, for example `MuJoCo in WSL -> ROS bridge in container -> MC in container -> telemetry observer`. Mark which component owns physics, timestamps, commands and readiness. Container creation and an RPC acknowledgement do not demonstrate that a command was applied.

## 2. Resolve clocks and Reset first

For each relevant timestamp, state whether it is ROS/simulation time, system wall time or a local monotonic clock. Do not subtract values from different clock domains or assume cross-host monotonic origins are comparable. A one-way latency measurement needs an established clock relationship; otherwise report local intervals or a clearly identified round trip, not an invented one-way delay.

In the running ROS environment, inspect the `/clock` publisher(s) and `use_sim_time` of the affected nodes. For example, use `ros2 topic info --verbose /clock` and `ros2 param get <node> use_sim_time` with actual node names. Verify the timestamp used by the implementation, not just parameter presence. With active ROS time, zero can mean the clock has not initialized; pause can legitimately leave simulation time unchanged. A reset/backward jump invalidates pre-reset caches and commands: inspect the installed reset handling and identify any missing generation/reset protection without claiming the proposal is already deployed. See [ROS clock design](https://design.ros2.org/articles/clock_and_time.html).

Use steady/monotonic elapsed time for local watchdogs and observation windows. Compare simulation progress with monotonic elapsed time to measure real-time factor; GUI frame rate is a different measurement. Check integration/control `dt`, message acquisition stamps and receipt times independently. Wall-clock NTP/PTP synchronization cannot repair a paused `/clock`, mixed `use_sim_time` or stale pre-reset TF.

## 3. Separate discovery from data delivery

Check endpoint names/types, domain and discovery scope, selected RMW, interface configuration and publisher/subscriber QoS. Inspect actual endpoint policies using the installed CLI/API. Discovery can succeed while samples fail to arrive. QoS incompatibility can prevent delivery; switching every stream to reliable can also add retries/queueing instead of improving freshness. Use the [Humble QoS contract](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst), not a newer distribution's assumed defaults.

For cross-boundary failures, establish these conditions before proposing a configuration change:

- **WSL:** distinguish NAT from mirrored mode and identify Windows/Hyper-V firewall or VPN involvement. Mirrored mode supports multicast on supported Windows/WSL versions; this is not proof it is enabled or reachable on this machine. Consult [Microsoft's WSL networking guide](https://learn.microsoft.com/en-us/windows/wsl/networking).
- **Docker:** Linux Engine host networking shares the daemon host's network namespace; it does not automatically unify Windows, WSL and all containers. Docker Desktop host networking has version/enablement and layer-4 limitations. Host mode is one option, not a DDS requirement; bridge discovery depends on topology, routing and DDS discovery configuration. See [Docker host networking](https://docs.docker.com/engine/network/drivers/host/).
- **Shared memory:** distinguish available, configured and actually used transports. Check vendor/version support, namespace visibility, permissions and `/dev/shm` limits only when logs or measurements implicate SHM. `ipc: host` is not a general DDS fix and shares an IPC namespace; do not enable it incidentally.

Compare endpoints within one boundary before crossing another, using existing authorized traffic or a bounded non-actuating diagnostic. Do not disable firewalls, replace RMWs, enable privileged mode, change network/IPC mode or alter controller deadlines as an unmeasured first response.

## 4. Distinguish loss, queueing and observer failure

Follow a saved sample from production through transport, callback and log. Check producer rate, receipt/callback rate, latest-value replacement, queue depth and observer/log capacity. A logger reaching its record cap while physics continues is a telemetry failure, not evidence that the robot stopped. A sequence gap may represent intentional coalescing, filtering, reordered samples or a late subscriber; do not call it network packet loss without boundary-specific evidence.

Within an identified stream and reset generation, count produced/received samples, duplicates, reordering, coalescing and confirmed missing samples separately. Define start/end alignment and allow in-flight data to drain before final loss accounting. If vendor messages lack sequence/generation fields, use existing correlated logs or a project-owned diagnostic sidecar; do not silently change vendor wire formats. State any remaining ambiguity.

Measure only the unresolved path. Record workload/model identity, warm-up and measurement window, clock domain/mapping, producer/consumer rates, latency distribution (p50/p95/p99/max with sample count), jitter definition, queue high-water mark, missed deadlines and CPU/memory overhead. `ros2 topic hz` measures what that observer receives and can itself be limited by QoS or load; it does not establish publisher frequency or end-to-end latency by itself.

## 5. Investigate scheduling only with evidence

If clocks, endpoint compatibility and queues do not explain the failure, inspect callback/executor duration, CPU throttling, memory pressure, rendering/recording overhead and host load. Compare the same bounded workload and avoid assuming WSL2 provides hard real-time scheduling. Do not add real-time priorities, CPU pinning or more threads before identifying contention; do not tune physics `dt` or safety thresholds to hide delayed telemetry.

When trace detail is necessary, check the installed ROS tracing support first. [Humble ros2_tracing](https://github.com/ros2/ros2_tracing/tree/humble) uses LTTng on Linux; ROS user-space events require LTTng-UST, not necessarily kernel tracing. Kernel scheduling analysis is a separate capability check for the actual WSL kernel/container permissions. A build with tracing disabled or LTTng absent can produce no events. Empty traces do not prove callbacks never ran. Keep captures bounded and record instrumentation overhead and trace-buffer losses. Installation, kernel/module changes and starting a new robot episode are separate actions, not prerequisites for reading an existing trace.

## Report and choose one correction

Return the first failing boundary, expected versus observed behavior, clock domains, exact version/configuration, evidence location and one smallest justified correction. Name unmeasured quantities explicitly. Preserve trial budgets and model/safety contracts. A documentation/static check qualifies this diagnostic guidance only; actual time synchronization, DDS delivery and latency need an authorized runtime check.

## Reference provenance

Reviewed 2026-11-15. This project-specific checklist is an adaptation, not an official ROS or Docker Skill. Primary documentation above governs platform claims; verify it against the installed release before applying it.

- [robotics-agent-skills: Docker ROS 2 development](https://github.com/arpitg1304/robotics-agent-skills/blob/f9bc5467ff9ee3d23f1a1b0b29a649843bb6ad11/skills/docker-ros2-development/SKILL.md), Apache-2.0, reviewed at `f9bc5467ff9ee3d23f1a1b0b29a649843bb6ad11`: container/DDS boundaries. Its host/bridge networking examples are conditional; no wholesale installation or execution is implied.
- [Robium integration Skill](https://github.com/robium-ai/robium/blob/ce944953d11ccdbd32613c4746389bc78a24e8c5/skills/integration/SKILL.md), MIT, reviewed at `ce944953d11ccdbd32613c4746389bc78a24e8c5`: explicit time, backpressure, ownership and health at module boundaries. No new application protocol or container split is required.
