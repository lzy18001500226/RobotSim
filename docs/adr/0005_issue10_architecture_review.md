# ADR-0005: Issue #10 Architecture Review Evidence

Status: Proposed (maintainer architecture review required; not accepted)
Evidence date: 2026-10-04
Related: Issue #10, PR #3, ADR-0004

## Purpose and Scope

This is the bounded Issue #10 evidence update for the proposed physics topology.
It compares the standalone pinned MuJoCo backend with Unity-hosted direct-load
MuJoCo. It does not select a transport, accept ADR-0005, authorize a merge, or
replace the physics/Unity/ROS 2 responsibility split in accepted ADR-0004.

The Unity candidate tested here directly loaded the original G1 MJCF and retained
native `mjModel`/`mjData`. It did not use `MjImporterWithAssets`, regenerated
MJCF, or automatic `MjScene` reconstruction. The earlier importer-generated
model was not comparable because mesh topology and 29 sensors differed; that
result does not describe this direct-load candidate.

## Matched Identity

| Component | Verified identity |
|---|---|
| Unitree source | `unitree_mujoco` commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` |
| Model | Original `g1_29dof.xml`; SHA256 `423e28bd718b19f7a65cda539b6f794ddbb268b4b9bdbd85f4bd982b30729617` |
| Meshes | All 60 referenced STL hashes matched the retained manifest |
| Unity | Editor `6000.3.25f1`, revision `e1dba0a9aba4` |
| Unity MuJoCo package | 3.3.6, source commit `eacad44a1a67afe520b263c9b15dab82f62a10aa` |
| Native MuJoCo | 3.3.6; standalone and Unity plugin SHA256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495` |
| Benchmark trace | Canonical 1,000-row trace; SHA256 `0c297cc4d059e339985ca45d9ea1e01a4e87250657baee63101c6306536e4f6f` |
| Timestep | `0.002 s` |

## Evidence Summary and Measurements

### Exact model and replay parity

Direct-load structural equivalence **passed**. Exact discrete structure and
ordering matched; maximum compiled numeric fingerprint delta was 0 against the
`1e-12` comparison threshold. Counts were `nq=36`, `nv=35`, `nu=29`, `nbody=31`,
`njnt=30`, `ngeom=73`, `nsite=2`, `nsensor=95`, and `nsensordata=113`. All 29
`jointactuatorfrc` sensors matched. Geometry matched at 36 meshes, 185,482
vertices, 370,714 faces, and 1,112,142 indices. Timestep, solver/integrator and
options, actuator ranges/gains, collision properties, contact pairs, and
exclusions matched.

The canonical 1,000-step replay matched simulation time, control, qpos, qvel,
actuator force, and sensor data exactly (maximum absolute delta 0). The trace
produced zero contacts in both runs; contact counts and canonical pair sequences
were exactly equal. This proves parity for the pinned G1 model and tested trace,
not for every model or workload.

### Matched performance

Ten alternating matched pairs ran for 30 seconds each after a consistently
excluded five-second warmup. Per-run contention gates passed. Both candidates
used the same byte-identical MuJoCo 3.3.6 native library, original model and
meshes, actuator trace, and 2 ms timestep. Unity ran in Editor batch mode with
rendering, transforms, and a managed state mirror disabled.

| Metric | Standalone | Unity direct-load |
|---|---:|---:|
| Mean throughput | 19,928.312 steps/s | 17,821.685 steps/s |
| Median trial throughput | 20,081.528 steps/s | 18,313.594 steps/s |
| Trial throughput p95 / p99 | 21,030.106 / 21,030.106 steps/s | 19,527.587 / 19,527.587 steps/s |
| Trial throughput CV | 3.469% | 6.708% |
| Trial range; `(max-min)/mean` | 18,764.769-21,030.106; 11.367% | 15,727.274-19,527.587; 21.324% |
| Pooled step-time p50 / p95 / p99 | 45.763 / 70.057 / 124.950 us | 49.200 / 86.200 / 146.400 us |
| Pooled p95-p50 step-time spread | 24.294 us | 37.000 us |
| CPU, one-core equivalent | mean 99.906% | mean 101.082% |
| Mean process RSS | 447.76 MiB | 836.48 MiB (includes Unity Editor) |

Unity throughput was lower in all ten pairs; ratio-of-means penalty was 10.571%.
Across 20,890 Editor update callbacks, `EditorApplication.update` work had
p50/p95/p99 13.497/18.004/22.414 ms. The residual beyond measured native steps
had mean 62.314 us and p50/p95/p99 41.8/135.3/197.9 us; it was 0.443% of
measured callback work.
That residual is not full Unity integration overhead: rendering, transform
synchronization, and managed state copying were disabled. Trial-rate p95/p99
have only ten observations and both equal the largest trial; pooled per-step
percentiles include correlated adjacent steps. These are valid matched local
hot-path measurements, not production headroom or rendered-FPS evidence.

### Timestamped state and buffering

Three ten-second fixture trials per topology requested a 500 Hz producer and
60 Hz consumer, used an eight-slot queue, and injected a 150 ms producer pause.
Both queues reached but never exceeded capacity. Standalone publication-to-
consumer availability p50/p95/p99 was 0.984/1.908/2.006 ms and snapshot age was
0.997/1.931/55.573 ms. Unity direct-load availability was 1.008/1.913/1.996 ms
and snapshot age was 1.025/1.939/54.729 ms. The age p99 includes the deliberate
pause. Standalone recorded 729 overflow drops and 12,278 consumer coalesces;
Unity recorded 606 drops and 12,391 coalesces. Each had 18 stale ticks. Drop
and coalesce counters reconciled with sequence gaps. These fixtures used
in-process queues; they are not IPC or DDS transport measurements.

In a separate Unity marker test, the retained native `mjData->time` and
`(generation, step_id)` were propagated to Unity. All 1,082 normal Update reads
matched the worker's current generation, step, and simulation time. At the CPU
`OnPostRender` callback, state lag p50/p95/p99 was 2/5/6 steps (4/10/12 ms),
maximum 8 steps (16 ms). Publish-to-Update wall latency was
0.906/1.917/2.196 ms; step-complete-to-`OnPostRender` was
5.644/10.518/12.972 ms. `OnPostRender` is not GPU completion or display
presentation. Pause held the same simulation identity and stale input was
detected. On reset, generation 2 step 0 was coalesced before Unity read it; the
generation change was later observed at step 1.

The standalone fixture measured local consumer availability and age, but did
not instrument a separate Unity renderer consuming standalone backend state.
Thus standalone-to-Unity wall latency and simulation-time alignment remain
unknown.

### Failure isolation and restart

In disposable processes, the standalone backend owned `mjModel`/`mjData`; Unity
only observed a small status snapshot. Killing Unity left the backend stepping
(step 5,225 to 9,250); restarting Unity reattached to the same backend state.
Killing the backend left Unity alive to report stale state. Restarting the
backend created a new generation from initial model state; the saved status
snapshot was not an `mjData` checkpoint.

For direct-load, the Unity process owned both frontend and native physics state.
Killing Unity ended stepping and lost the live state. Relaunch loaded the model
at step zero. No independent physics restart or checkpoint restore was tested.
These are bounded scratch process fixtures, not deployed service supervisors.

## Tradeoff Matrix

Confidence: **High** means directly measured against the pinned setup;
**Medium** means a repeatable scratch fixture with important integration limits;
**Low** means an inference or an unmeasured path. Each topology cell states
evidence, confidence, and remaining uncertainty.

| Criterion | Standalone backend | Unity-hosted direct-load |
|---|---|---|
| Exact model parity | **Evidence:** Pinned native load is the reference side of exact structural and replay comparison; numeric delta 0, 95 sensors/113 values, all 29 actuator-force sensors and mesh counts match. **Confidence:** High. **Uncertainty:** Only pinned G1 29-DoF and tested trace. | **Evidence:** Original MJCF loaded directly into retained native model/data; structural and 1,000-step replay parity exact. **Confidence:** High. **Uncertainty:** Other models, configurations, and application workloads not tested. |
| Physics authority | **Evidence:** Backend process owned model/data and continued stepping through Unity frontend termination. **Confidence:** High for fixture. **Uncertainty:** Production backend service/API not exercised. | **Evidence:** Unity process owned model/data; marker sync was read-only. Unity termination ended physics. **Confidence:** High for scratch host. **Uncertainty:** Production Unity lifecycle and scene integration untested. |
| Performance | **Evidence:** 10-pair mean/median 19,928/20,082 steps/s; pooled step p50/p95/p99 45.763/70.057/124.950 us; CV 3.469%, CPU 99.906%, RSS 447.76 MiB. **Confidence:** Medium-high for matched hot path. **Uncertainty:** No deployed workload, ROS 2, Unity rendering, or headroom result; p95/p99 across ten trial rates are coarse. | **Evidence:** 10-pair mean/median 17,822/18,314 steps/s; pooled step p50/p95/p99 49.200/86.200/146.400 us; CV 6.708%, CPU 101.082%, Editor RSS 836.48 MiB; 10.571% ratio-of-means throughput penalty. **Confidence:** Medium-high for batch Editor hot path. **Uncertainty:** No full scene, transform sync, rendering, or production integration overhead; RSS includes Editor. |
| Latency / jitter | **Evidence:** In-process local consumer availability p50/p95/p99 0.984/1.908/2.006 ms; age 0.997/1.931/55.573 ms. **Confidence:** Medium for fixture. **Uncertainty:** No backend-to-separate-Unity IPC latency/jitter. | **Evidence:** Fixture availability 1.008/1.913/1.996 ms; marker publish-to-Update 0.906/1.917/2.196 ms; CPU render callback handoff 5.644/10.518/12.972 ms. **Confidence:** Medium for scratch fixtures. **Uncertainty:** No transport, GPU/display timing, or production workload. |
| Buffering | **Evidence:** Fixed queue cap 8, never exceeded; 729 drops, 12,278 coalesces, 18 stale ticks across three trials. **Confidence:** High for queue fixture. **Uncertainty:** No production IPC/DDS QoS/backpressure test. | **Evidence:** Same cap, never exceeded; 606 drops, 12,391 coalesces, 18 stale ticks; reset step 0 was coalesced in latest-state handoff. **Confidence:** High for fixture semantics. **Uncertainty:** No selected transport or critical-event delivery guarantee. |
| Failure isolation | **Evidence:** Unity frontend termination did not stop backend; backend death left Unity alive to report stale state. **Confidence:** High for disposable processes. **Uncertainty:** Test frontend used a status shim, not production Unity/ROS 2. | **Evidence:** Unity termination stopped both frontend and physics owner. **Confidence:** High for process boundary. **Uncertainty:** Managed fault modes and recovery service not tested. |
| Restart behavior | **Evidence:** Unity restart reattached without physics reset; backend restart loaded initial state under new generation, with no checkpoint. **Confidence:** High for fixture. **Uncertainty:** Supervisor, state restore, and command replay not tested. | **Evidence:** Unity restart loaded model at step 0 and lost old live state. **Confidence:** High for scratch host. **Uncertainty:** Explicit checkpoint/restore could change behavior but is unimplemented. |
| Unity complexity | **Evidence:** Requires a backend lifecycle, state/command bridge, clock mapping, and buffering; tests used a status file and in-process queue only. **Confidence:** Low-medium inference. **Uncertainty:** No maintained production bridge or effort comparison. | **Evidence:** Direct-load parity works without importer conversion; one read-only marker transform was synchronized. **Confidence:** Medium for feasibility. **Uncertainty:** Plugin/binding distribution, threading, full rig, scene, renderer, and sensor integration effort not measured. |
| Timestamp alignment | **Evidence:** Local fixture carries step/time metadata and measures consumer availability/age. **Confidence:** Medium for fixture. **Uncertainty:** Separate standalone-to-Unity rendering path is unmeasured. | **Evidence:** 1,082 normal Update reads matched generation/step/sim-time; render callback lag p50/p95/p99 2/5/6 steps; pause and stale input exercised. **Confidence:** High for one-marker scratch harness. **Uncertainty:** Reset step zero was not applied; no GPU/display timestamp or full robot rig. |
| ROS 2 implications | **Evidence:** Process boundary can keep backend state independent of Unity and permits a ROS 2-facing owner/adapter. **Confidence:** Low-medium inference. **Uncertainty:** No SDK2/DDS round trip, ROS 2 graph, QoS, or clock synchronization tested. | **Evidence:** Unity can host or connect a ROS 2 client/bridge, but shares physics process fate. **Confidence:** Low inference. **Uncertainty:** No client/bridge, DDS, deployment, or timestamp/QoS integration tested. |
| Sensor implications | **Evidence:** MuJoCo native sensor inventory/replay matches, including 29 actuator-force sensors. **Confidence:** High for native sensors. **Uncertainty:** Unity RGB/depth/LiDAR output, timestamps, transport, and fusion untested; Unity-originated data needs an explicit path. | **Evidence:** Same native sensor inventory and data preserved; marker alignment only. **Confidence:** High for native sensors, low for external sensors. **Uncertainty:** No rendered RGB-D/LiDAR acquisition/publication or latency test; colocation is only a possible convenience. |

## Proposed ADR-0005 Wording

> **Status: Proposed; maintainer architecture review required.** For the initial
> RobotSim baseline, prefer a standalone MuJoCo/Unitree backend as the leading
> physics-authority topology. The backend owns `mjModel`, `mjData`, simulation
> time, physics stepping, and the backend-facing low-level control path. Unity
> consumes timestamped state and owns rendering, external sensor simulation,
> and visualization; ROS 2 remains the system-integration boundary. This
> recommendation is supported by process failure isolation and restart
> behavior, plus better throughput, step-time distribution, trial spread, and
> process RSS in the matched local benchmark.
>
> Unity-hosted direct-load remains technically viable: on the pinned G1 29-DoF
> MJCF and MuJoCo 3.3.6 pair it preserved exact compiled model, sensor,
> geometry, and tested replay equivalence without the lossy importer round-trip.
> Keep it available for bounded Unity-centric use; do not treat the importer
> generated model as equivalent for this G1. The measured benchmark does not
> establish production performance headroom. Select no transport or rate
> guarantee in this ADR. Maintain one MuJoCo physics authority and do not
> reconstruct robot physics from Unity transforms.

## Unresolved Risks and Maintainer Gate

- Validate the production Unitree SDK2/DDS LowCmd/LowState round trip and audit
  shared `mjData` access/thread safety; this review's process fixture was not
  the production vendor bridge.
- Measure end-to-end state latency, jitter, backpressure, and recovery using a
  real standalone-to-Unity process boundary. Current queue timings are
  in-process and do not select transport or QoS.
- Validate full G1 visual transforms and scene behavior in Unity, plus actual
  RGB/depth/LiDAR acquisition timestamps and ROS 2 publication. The timestamp
  test used one marker; its render callback was CPU-side only.
- Define reset/generation delivery semantics: the latest-state slot coalesced
  generation 2 step zero in the measured run.
- Verify restart/checkpoint expectations and service supervision for the
  selected deployment.
- Validate high-concurrency and target deployment headroom. The 10-pair result
  is a matched local harness result; it is not evidence for rendering,
  production bridge overhead, or RL scaling.
- Confirm compatible distribution/licensing for all model meshes and SDKs;
  the existing BSD-3-Clause finding applies only to the pinned Unitree source.

**Maintainer architecture gate:** RobotSim maintainers must review and explicitly
accept, revise, or reject ADR-0005. This PR only prepares that review. ADR-0005
remains Proposed; do not merge PR #3 or treat this recommendation as accepted.

## Evidence Preservation

Raw commands, runtime identity, mesh hashes, native structural comparison,
replay outputs, trial CSV/JSON/logs, queue traces, failure-process logs, and
timestamp traces are retained outside the repository under
`C:\Users\HP\Desktop\Robot\issue10-g1-equivalence\`. In particular, see
`direct-load/`, `performance/20261004-reproducibility-10pairs/`,
`state-flow/20261004-bounded-snapshot/`, `failure-isolation/20261004/`, and
`timestamp-alignment/20261004/`. The raw benchmark directory is approximately
407 MiB and is not committed; this file records the identities and summarized
results needed for review.
