# ADR-0005: Physics Process Topology

Status: Proposed (maintainer architecture review required; not accepted)
Proposed: 2026-10-03
Evidence updated: 2026-10-04

## Problem

RobotSim needs one authoritative MuJoCo physics state while supporting Unitree G1 first, AgiBot X2 second, Unity rendering and external sensors, and ROS 2 integration. This ADR compares whether MuJoCo should run inside Unity or in a standalone backend. It does not select a transport or implement either production integration.

## Evaluated Architectures

### A. Unity-hosted direct-load MuJoCo

Unity owns the process lifecycle and hosts the native MuJoCo runtime. The Unity process loads the original MJCF directly, retains `mjModel` and `mjData` as the only physics state, and may copy read-only model state to Unity transforms for visualization. It does not reconstruct physics from transforms.

This candidate is distinct from the package's component/importer workflow, which converts between MJCF, Unity components, and generated MJCF. The pinned G1 importer round-trip was not equivalent; the Issue #10 direct-load test bypassed that path and passed structural and replay parity.

### B. Standalone MuJoCo backend

A separate MuJoCo process owns `mjModel`, `mjData`, physics stepping, simulation time, and the vendor-facing low-level control path. Unity consumes timestamped state and owns rendering, sensors, and visual interaction. ROS 2 remains RobotSim's system-integration boundary. No transport is selected here.

Both preserve the existing ADR-0004 responsibility split: MuJoCo is physics truth, Unity is the high-fidelity frontend, and ROS 2 integrates robot and application interfaces. The Issue #10 recommendation remains proposed; it does not select a transport or change ADR-0004.

## Authoritative Source Findings

### MuJoCo Unity plug-in

The repository Unity project at `apps/unity/project` is a Unity 6.3 template and its `Packages/manifest.json` does not currently configure `org.mujoco`. The compatibility experiment therefore used an isolated Unity project and an explicit official pair:

- Plug-in source: `google-deepmind/mujoco`, tag `3.3.6`, commit `eacad44a1a67afe520b263c9b15dab82f62a10aa`; `unity/package.json` also declares package version `3.3.6`.
- Native library: Linux `libmujoco.so.3.3.6` from the official MuJoCo 3.3.6 release, copied to the package as `libmujoco.so` as the official Unity guide directs. The package copy and release-extracted library have the same SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Runtime identity: the passing Unity test calls the plug-in's P/Invoke `mj_version()` and asserts `336`, which is MuJoCo 3.3.6. Thus the library loaded by that test, rather than only a nearby file, is verified as 3.3.6.

The official Unity guide explicitly recommends a version-specific MuJoCo tag and the corresponding release library; it warns that `main` may not match a release binary. At tag 3.3.6, `MjScene.CreateScene()` generates MJCF, compiles it to `mjModel`, creates `mjData`, and binds components. `FixedUpdate()` calls `StepScene()`; without a control callback, `StepScene()` calls `mj_step()` and then synchronizes active components. `MjBody.OnSyncState()` copies `xpos` and `xquat` into its Unity transform. The plug-in exposes `Model` and `Data` pointers to C# code, so in-process state access is possible.

The importer is separate from runtime stepping: the Editor importer uses MuJoCo to validate and save MJCF, then maps the saved model into Unity components and assets. Unity's scene hierarchy is therefore also the plug-in's model authoring representation. Unity owns the fixed-step schedule; the guide says the plug-in reads the simulation step from Unity's Fixed Timestep.

### Unitree and AgiBot upstreams

The inspected Unitree source is `unitreerobotics/unitree_mujoco` commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`. Its repository license at that commit is BSD 3-Clause; this finding applies to that repository, not automatically to every SDK, model, or asset used with it.

The C++ simulator owns `mjModel* m` and `mjData* d`, loads the model, creates data, and advances `mj_step()` in its physics thread. Its SDK2 bridge receives the same pointers. For G1, the bridge uses Unitree's `unitree_hg` low-level IDL, consumes `rt/lowcmd`, writes actuator controls using torque plus proportional/derivative terms, and publishes `rt/lowstate` motor and IMU state. Its recurrent bridge thread is configured at 1000 Hz. The Python simulator likewise constructs `MjModel`/`MjData` and calls `mj_step()` in its simulation thread; its default `SIMULATE_DT` is 0.005 s (200 Hz) and viewer interval is 0.02 s (50 Hz).

In the inspected Python and C++ paths, the bridge accesses MuJoCo arrays while the physics loop also accesses the same `mjData`; no common data lock is evident around all those accesses. Treat data-race behavior as an upstream integration risk to audit, not as a claim that the bridge is thread-safe. Standalone topology preserves the process and API boundary, but does not remove this risk by itself.

The current official AgiBot X2 simulation guide describes `sim_mujoco` as the MuJoCo physics/HAL process that subscribes to joint commands and publishes joint and IMU state, with the separate `mc` process producing commands. The guide presents this as one of two distinct simulation pipelines and says not to drive the same simulation through both. This is evidence that a standalone physics/HAL boundary is compatible with the X2 stack; it is not an X2 model or runtime validation in RobotSim.

## Initial Local Experiment Evidence (2026-10-03)

### A. Matched Unity 3.3.6 pair

Unity Editor `6000.3.25f1` ran an isolated PlayMode test in batch mode. The test generated a minimal one-hinge scene from official Unity plug-in components and verified this chain:

`Unity 6.3 -> MuJoCo Unity plug-in 3.3.6 -> loaded libmujoco 3.3.6 -> MjScene MJCF compilation -> mj_step() -> mjData/qpos change -> Unity Transform change`

The test asserts native version code `336`, compiled `nq == 1`, positive simulation time after two Unity fixed updates, a `qpos` change greater than `1e-5`, and a body Transform rotation change greater than `1e-3` degrees. Unity Test Framework reported 1 PlayMode test passed, 0 failed. The run used `-nographics`, so it proves numerical Transform synchronization, not visual appearance or sensor rendering.

The passing test used the unmodified official tag source and the matched release library. The canonical project remains free of a MuJoCo package copy; the test harness and native library stayed under `/tmp`. A previous MJCF-import diagnostic involved a generated `<flag passive>` field. A temporary diagnostic source edit was reverted; the restored `MjGlobalSettings.cs` SHA-256 matches the file at the official tag. No generated-field deletion or rewrite is part of this ADR or the passing test. The cause of that separate importer-fixture failure remains unresolved; it does not negate the successful minimal runtime chain. Because the 3.3.6 pair passed the required chain, no additional release tags were tested.

The Unity log contained one `[Licensing::Module] Error: Access token is unavailable; failed to update` entry and no warning entries. The PlayMode test nevertheless completed and passed; this validates the local test path only, not authenticated Unity services.

### B. Standalone G1 model stepping

Using the official Unitree G1 `g1_23dof.xml` and `g1_29dof.xml`, an isolated Python site-package containing MuJoCo 3.3.6 compiled each model, created `MjData`, applied a constant `ctrl[0] = 1.0`, and ran 1000 `mj_step()` calls. Both runs reported `nq=36`, `nv=35`, `nu=29`, `nsensor=95`, model timestep `0.002` s, and simulation time approximately `2.0` s. The named `left_hip_pitch` joint position changed from `0.0` to approximately `2.881` rad and `2.854` rad respectively; the resulting qpos values were finite. Actuator 0 is named `left_hip_pitch`.

This is a model-load and stepping check with an open-loop input, not a stable walking, controller-fidelity, real-time, or Sim2Real result. The official SDK2/DDS bridge was not run: the WSL environment lacks the SDK2 Python package, and the C++ build prerequisites include CMake and Unitree SDK2, which are not installed. Bridge behavior and message paths above are from pinned upstream source inspection.

## Initial Comparison (before Issue #10 matched measurements)

The following table records the initial source and design expectations. Measured
Issue #10 results and their limitations are in
[`0005_issue10_architecture_review.md`](0005_issue10_architecture_review.md);
use that evidence where it updates an earlier estimate or unmeasured claim.

| Criterion | A. Unity-hosted | B. Standalone backend |
|---|---|---|
| Setup complexity | Pin the Unity package source and copy the matching native library into the package; both steps are straightforward but version-coupled. | Official C++ setup also needs MuJoCo, CMake, Unitree SDK2, Boost, and GLFW; Python setup needs MuJoCo, Unitree SDK2 Python, CycloneDDS, and viewer dependencies. These SDK dependencies are absent in the current WSL environment. |
| Unitree upstream reuse | MuJoCo is proven to run in Unity, but the C++ bridge's direct `mjModel*`/`mjData*` contract would need a native in-process adapter, lifecycle integration, and synchronization with Unity stepping. | Best reuse: keep the official simulator and SDK2 bridge as a separate backend; add a RobotSim state-output adapter. DDS behavior still needs a local runtime test. |
| AgiBot X2 | Technically possible, but would need to adapt X2's process-oriented HAL/control boundary into Unity. | Closely matches the documented `sim_mujoco` HAL plus separate `mc` process topology. |
| 500-1000 Hz physics/control | Can configure Unity's fixed timestep to 1-2 ms, but scheduling is tied to Unity's player loop and catch-up behavior. No sustained-rate benchmark was run. | Physics clock is independent of rendering. The inspected Unitree C++ bridge is 1 kHz and the tested G1 model timestep is 2 ms; these facts do not prove deadline performance on this host. |
| Timing and determinism | One application and one clock simplify local ordering; physics work competes with rendering and other Unity work. | Physics/control owns its clock; snapshots must carry simulation timestamps and Unity must interpolate or select a sensor-time state. |
| Unity RGB/depth/LiDAR | Most direct access to Unity scene, cameras, and rendered objects. | Unity still owns cameras and sensors, but robot and dynamic-object poses must arrive across a process boundary. |
| MuJoCo/Unity synchronization | In-process state access and official component-to-Transform synchronization; no IPC serialization. | Adds snapshot production, mapping, buffering, and interpolation; bandwidth and latency were not measured. |
| ROS 2 / Sim2Real | Unitree DDS inside Unity would add native dependencies, threads, and vendor lifecycle to the Editor/player. | Keeps vendor bridge outside Unity and allows a separate RobotSim ROS 2 adapter; Unitree DDS is not itself the RobotSim ROS 2 contract. |
| Geometry source of truth | MuJoCo remains the physics authority, with Unity components also defining the plug-in's model hierarchy. | MuJoCo remains physics authority; Unity mirrors visual geometry and consumes state. A shared model/asset pipeline is still needed to prevent visual drift. |
| Headless tests and RL | Requires Unity runtime/player participation for this integrated physics path; feasibility and scaling were not tested. | Backend can run without Unity rendering and can be exercised independently; throughput and parallel scaling remain unmeasured. |
| Debugging | One scene is convenient to inspect, but native physics failure is inside the Unity process. | Separate logs and process debugging are cleaner boundaries, with timestamp correlation and startup supervision as added work. |
| Version coupling | Strict plug-in/native version match is required. The official 3.3.6 pair is now proven on Unity 6.3. | MuJoCo remains pinned within the backend; Unity can consume a versioned state schema without loading MuJoCo's native ABI. |
| WSL2 | The matched package loaded in Unity 6.3 under WSL2; the test was headless and does not validate rendered GPU sensors. | A Linux backend is natural in WSL2; the G1 model ran locally, but the full SDK2 executable was not built. |
| RobotSim-specific glue | Medium-high: native SDK2/DDS integration, thread/lifecycle management, and coordination with Unity-owned stepping. | Medium: snapshot producer/consumer, name/frame mapping, clock alignment, and a supervised backend process. No transport is chosen. |

### State synchronization and rate estimate for B

The initial snapshot should contain:

- simulation timestamp (and a monotonically increasing sequence number);
- robot root pose;
- named joint positions, with velocities optional for interpolation;
- poses of dynamic non-robot objects that Unity must render or use for external sensing.

If Unity's robot rig cannot reproduce MuJoCo link transforms exactly from root pose and joint positions, add named link poses rather than inventing a second physics representation. Use explicit frame and quaternion conventions. Keep control and simulation stepping in the backend; Unity should consume snapshots, not feed its render cadence back into the physics clock.

Starting estimates, not measurements: 500 Hz physics for a 2 ms model step; up to 1 kHz vendor low-level bridge where the robot contract requires it; 60-120 Hz snapshots for a 30-60 Hz render/sensor frontend, with interpolation against simulation timestamps. Sensor acquisition rates remain sensor-specific. Do not choose shared memory, UDP, ROS 2, Zenoh, or another transport until latency, deployment, and failure-handling requirements are measured.

## Current Proposed Recommendation

**Proposed only; not accepted.** Prefer B as the leading RobotSim v1 baseline:
standalone MuJoCo owns `mjModel`, `mjData`, simulation time, physics stepping,
and the backend-facing low-level control path; Unity consumes timestamped state
and owns rendering, external sensor simulation, and visualization; ROS 2
remains the system-integration boundary. This proposal is supported by the
standalone process surviving Unity frontend termination, independently
restarting, and performing better in the matched local hot-path benchmark.

Keep A technically viable for bounded Unity-centric use: direct loading of the
original pinned G1 model preserved exact structural, sensor, geometry, and
tested replay equivalence without the importer round-trip. The matched local
benchmark does not establish production headroom; the measured queue fixtures
do not select IPC/DDS transport or QoS. Do not reconstruct robot physics from
Unity transforms. The complete measurement matrix and proposed wording are in
[`0005_issue10_architecture_review.md`](0005_issue10_architecture_review.md).

## Consequences

- MuJoCo is the single source of physical state; Unity does not run a second robot physics simulation.
- Unitree SDK2/DDS can remain in a backend process, but its shared-data threading behavior must be reviewed and the RobotSim ROS 2 interface still needs an adapter.
- Unity needs timestamped pose/joint ingestion, interpolation, stable name/frame mapping, and process supervision.
- Visual/collision geometry synchronization remains an explicit asset-pipeline responsibility.
- Unity-hosted direct-load remains a viable implementation path, but physics state and frontend share the Unity process failure domain.
- No transport, production state schema, QoS policy, or rate guarantee is selected by this proposal.

## Unresolved Questions

- Does the pinned Unitree SDK2 bridge complete a G1 LowCmd/LowState round trip in this WSL2 environment, and does its shared `mjData` access need a synchronization fix?
- What state representation best preserves link transforms and Unity sensor alignment without duplicating physical state?
- What production transport/QoS, end-to-end latency, jitter, stale-state policy, and process-restart behavior meet the intended RGB-D/LiDAR workloads?
- How should generation changes/reset samples be delivered when a latest-state queue can coalesce step zero?
- Which ROS 2 and vendor DDS boundaries are required for the X2 backend and real-robot parity?
- Does the measured standalone performance preserve enough headroom under rendered Unity, ROS 2, sensors, and high-concurrency workloads?
- Are all model meshes and SDK dependencies covered by licenses compatible with the intended distribution? The BSD-3-Clause finding above is limited to the pinned `unitree_mujoco` repository.

## Validation Record

### Automatically validated

- Official MuJoCo Unity tag/commit and package version identified; package library hash matches the official 3.3.6 release library.
- The test's loaded native library reports version code 336.
- Unity 6.3 PlayMode: model compile, `mj_step`, changing qpos, and changing Unity Transform; 1 passed, 0 failed.
- Both official G1 MJCF variants compile with MuJoCo 3.3.6 and advance 1000 steps with finite, changing joint state.
- The original pinned G1 29-DoF MJCF matches Unity-hosted direct-load structurally and on the canonical 1,000-step replay; native sensor and geometry inventories match exactly.
- Ten matched standalone-vs-Unity-direct-load performance pairs completed with identical native/model/trace/timestep identity and passed their contention gates.
- Bounded snapshot, disposable process failure/restart, and Unity marker timestamp fixtures were measured; they do not establish a production transport or full-scene integration.
- Unitree C++/Python bridge paths, message topics, ownership, and configured rates inspected at the pinned commit; X2 process topology inspected in current official documentation.
- Unity-generated `Library/`, `Temp/`, `Logs/`, `obj/`, build output, and machine-local settings are ignored. Experiment-only plugin copies and harnesses are outside the repository.

### Manually validated

- None. The PlayMode run used `-nographics`; no human visual inspection, real robot, or physical sensor validation was performed.

### Not validated

- Unity's rendered appearance, camera/depth/LiDAR alignment, or GPU sensor performance.
- A full Unitree SDK2 Python or C++ process, DDS LowCmd/LowState exchange, controller behavior, or deadline performance.
- Production IPC/DDS transport latency/jitter and QoS/backpressure, rendered-scene performance headroom, headless/RL scaling, or production process supervision.
- A full standalone-backend-to-separate-Unity timestamp-alignment path, actual RGB-D/LiDAR acquisition/publication timing, or exact rendering of a reset step-zero state.
- AgiBot X2 model/runtime integration in RobotSim.

## References

- [MuJoCo 3.3.6 Unity plug-in guide](https://mujoco.readthedocs.io/en/3.3.6/unity.html)
- [MuJoCo 3.3.6 release](https://github.com/google-deepmind/mujoco/releases/tag/3.3.6)
- [MuJoCo Unity source at commit eacad44](https://github.com/google-deepmind/mujoco/tree/eacad44a1a67afe520b263c9b15dab82f62a10aa/unity)
- [Pinned Unitree MuJoCo source at commit 1eb6642](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d)
- [Pinned Unitree repository license](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/LICENSE)
- [AgiBot X2 Sim + MC deployment guide](https://x2-aimdk.agibot.com/en/latest/sim_index.html)
- [AgiBot simulation and RL pipeline overview](https://x2-aimdk.agibot.com/en/latest/sim_rl/index.html)
