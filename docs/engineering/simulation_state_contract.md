# Simulation state and control contract

**Status: Proposed for review.** This document defines topology-independent meaning for simulation state and latest-value commands. It supplements the [runtime contract](runtime.md) and [ADR-0004](../adr/0004_mujoco_unity_ros2_architecture.md); those documents remain authoritative for adopted units, frames, time use, and separation of simulation truth from estimates. This contract does not select a physics topology, process boundary, or transport.

## State envelope

Every state sample identifies its `schema_version`, `model_identity`, `generation` (epoch), per-stream `sequence`, `sim_time`, and lifecycle state. A sample is interpreted only within its model identity and generation.

- **Time:** `sim_time` is a simulation timestamp in seconds. It may stop while paused, but it must not move backward within a generation. A reset, rewind, or other time rollback is a discontinuity and **MUST** start a new generation before the producer publishes a sample at the rolled-back time. A generation change invalidates all cached state and pending commands from the previous generation. The first `sim_time` value and operation-specific reset/rewind behavior: **BLOCKED BY #10**. Use a steady/monotonic clock for local freshness, watchdogs, and elapsed wall time; never use simulation time for those deadlines or treat monotonic time as simulation time. Host-monotonic/backend-simulation mapping, whether producer and consumer clocks are directly comparable, and cross-process clock interpretation: **BLOCKED BY #10**.
- **Sequence and generation:** each state or command stream has its own sequence, increasing for every newly produced value within one generation and restarting at zero in a new generation. A sample is identified by its generation, stream, and sequence; sequence values from different generations are never compared. Sequence does not wrap within a generation. Consumers accept samples only for the currently established generation and discard samples tagged with any other generation, including delayed samples from a prior generation. Retransmission retains its original sequence. Exact field encoding, generation-transition signaling, and generation-token representation: **BLOCKED BY #10**.
- **Model and schema:** model identity names the robot and exact model revision; the schema version identifies the state layout and conventions. Bump the schema version whenever fields, ordering, or interpretation change; consumers accept only versions they understand. Include the ordered joint-name list with the schema, and make each joint array use that same order. Consumers map by name rather than infer an order or substitute zero for a missing joint. The model catalog/digest boundary: **BLOCKED BY #10**.
- **Root and joints:** root state contains pose and twist relative to explicitly named frames; for humanoids, `base_link` is waist-attached per the runtime contract. Joint state contains names and position; velocity and effort are included when available and otherwise marked unavailable. Use SI units and the body/world frame conventions already specified in the runtime contract. Quaternions are normalized and ordered `x, y, z, w`; a pose quaternion represents the child frame orientation in its named parent frame.
- **Links:** link state, if provided, is derived from the same model revision and state sample as root/joint state. The default transported link set and its measurement cost: **BLOCKED BY TRANSPORT BENCHMARK**.
- **Truth and estimates:** MuJoCo physical state is ground truth. Localization, perception, and other estimator outputs remain separately identified estimates; an estimate must not overwrite or masquerade as ground truth. Follow the channel and timestamp distinctions in the [runtime contract](runtime.md).

## Lifecycle and sample handling

| Producer event/state | Required semantics | Consumer action |
|---|---|---|
| `STOPPED` | No valid state or command is available. On start, validate model identity/schema before entering `READY`. | Mark state unavailable; do not keep cached state presented as current. |
| `READY` | Model identity and schema are known; simulation is not advancing. | Wait for a valid sample before exposing state. |
| `RUNNING` | Simulation time advances according to the backend; each new sample gets a higher sequence. | Accept only matching model/schema and current generation. |
| `PAUSED` | Physics and simulation time stop. A repeated snapshot may be emitted with a higher sequence and unchanged simulation time. | Track pause explicitly; equal simulation timestamps alone do not imply a duplicate. |
| Resume | Continue from the paused simulation time in the same generation. | Expire any command whose monotonic receipt-time lease elapsed during pause; require a fresh command before applying a new setpoint. |
| `RESETTING` | Invalidate pending commands and cached state; create a new generation, restart each stream's sequence at zero, then publish the initial state before returning to `READY`. The first simulation timestamp follows the reset/rewind operation's semantics (**BLOCKED BY #10**). | Discard all prior-generation state and commands; wait for the initial sample of the new generation. |
| Restart | A producer restart has the same invalidation effect as reset and creates a new generation before publishing state. Each stream's sequence restarts at zero. | Clear high-water sequence and caches on generation change; revalidate model/schema. |
| `FAILED` | Stop publishing valid state and report failure. Failure signaling mechanism: **BLOCKED BY #10**. | Mark state stale after its freshness limit; do not silently keep presenting it as current. |

The semantic transition is `STOPPED → READY → RUNNING ↔ PAUSED`; reset passes through `RESETTING → READY`, and restart returns to `READY` with a new generation, including recovery from `FAILED`. A manual single-step, if exposed, is valid only while paused, advances one configured integration step, and returns to `PAUSED`; whether that operation is exposed and how it maps to the selected backend: **BLOCKED BY #10**.

Within the active generation and stream, a consumer accepts a sequence greater than its last accepted sequence. It ignores equal or lower sequences as duplicate or reordered data. A sequence gap is counted as dropped/coalesced data; because a state sample is a snapshot, accept the newest valid sample rather than waiting for missing older samples. A sample whose `sim_time` moves backward while carrying the active generation is invalid: reject it, report the discontinuity/protocol violation, and do not silently accept it. The producer **MUST** establish a new generation before publishing post-rollback state. On an explicit generation transition, clear state and command caches and reject every sample not tagged with the new active generation. A sample older than the configured local-monotonic freshness limit is stale and must not be presented as current. Numeric state freshness and command-expiration thresholds: **BLOCKED BY TRANSPORT BENCHMARK**.

## Latest-value commands and bounded queues

- Commands are latest-value setpoints, not a history to replay. Per command source and generation, a higher sequence replaces any older pending value; reject duplicate or lower sequences. Reset and restart invalidate every command targeting the previous generation.
- The consumer retains at most one pending state snapshot and one pending latest-value command per source. When a newer value arrives, replace the older pending value and count the coalescing/drop. Do not build unbounded queues or block the simulation loop waiting for a consumer.
- A consumer starts a command watchdog from its local monotonic receipt time. When no valid command arrives within the configured timeout, expire the command and apply the command type's documented safe fallback (zero velocity for a velocity setpoint). Fallbacks for any other command type must be defined before use: **BLOCKED BY #10**. Numeric command-expiration thresholds: **BLOCKED BY TRANSPORT BENCHMARK**.
- Lifecycle operations such as reset, pause, resume, and step are not coalesced as state values. Permit at most one such operation in flight; acknowledge acceptance or reject it as busy/invalid, and make repeated requests identifiable so they cannot apply twice. The transport-level delivery/acknowledgement mechanism: **BLOCKED BY #10**.

## Mandatory performance telemetry

Every transport benchmark and performance report records the target machine/runtime, model identity and schema, workload, warm-up and measurement duration, producer/consumer rates, and:

- end-to-end latency distribution (p50/p95/p99 where meaningful), jitter, and missed deadlines;
- queue depth/high-water mark, coalesced/dropped/reordered/duplicate/stale samples, and rejected/busy operations;
- reset/restart recovery time and behavior when producer or consumer stops;
- CPU and memory overhead.

Report the measurement method and conditions with the result. Transport-specific physical buffer sizes and numeric throughput, latency, jitter, freshness, resource, and recovery thresholds: **BLOCKED BY TRANSPORT BENCHMARK**. Passing a functional or headless simulation check does not establish those performance bounds.

## Explicitly unresolved

- Physics/process ownership, clock-domain comparability and mapping (including host monotonic to backend simulation time and cross-process clock interpretation), generation-transition signaling, reset/rewind operation semantics, concrete transport/wire encoding, model catalog/digest boundary, and exposure of single-step: **BLOCKED BY #10**.
- Numeric freshness/timeout thresholds and measured transport budgets: **BLOCKED BY TRANSPORT BENCHMARK**.
- The measured cost/profile and default transported set for optional link state: **BLOCKED BY TRANSPORT BENCHMARK**.
