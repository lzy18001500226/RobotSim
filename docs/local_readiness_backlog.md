# Local Readiness and Development Backlog

Snapshot date: 2026-10-03

This is a readiness record and a set of issue drafts, not authorization to start
implementation. Production robot-runtime work did not start during this Goal.

## Task Baseline

- The starting spike/physics-ownership worktree was clean.
- origin was fetched; local main was fast-forwarded to origin/main
  (e701e93387d35eb18a7897c3c2a79f0c98d6a3f0).
- This Goal is on chore/local-readiness-backlog, created from that main.
- PR #3 remains open and review-gated. Its local and remote source branch still
  point to 8b60cc624d3655eeb4ef0b7de8516d7285e7b352; it was not modified or
  merged.

## Local Agent Infrastructure

| Check | Status | Evidence and boundary |
|---|---|---|
| .codex/hooks.json parses | PASS | python3 -m json.tool completed successfully. |
| Agent-infrastructure tests | PASS | python3 -m unittest discover -s tests -v: 12 tests passed, 0 failed. |
| Agent-script/test compilation | PASS | python3 -m py_compile completed for both scripts and both test files. |
| Notification dry-run | PASS | ready_for_review --dry-run formatted the notice and explicitly reported that no SMTP connection was made. No real email was sent. |
| Local Codex discovers this WSL project's hooks | MANUAL CHECK REQUIRED | The hook file is in the documented project location, but the WSL codex --version probe could not start: its wrapper reported node: not found. This does not establish what the desktop app currently loads. |
| Hook trust and approval state | MANUAL CHECK REQUIRED | Codex requires the project .codex layer to be trusted and requires review/trust of the exact current non-managed hook definition before it runs. The current UI trust state was not observable from this shell. See the [official hook documentation](https://learn.chatgpt.com/docs/hooks). |
| Repository works without project hooks | PASS | The test suite, JSON validation, compilation, and ordinary Git status checks ran directly in WSL without invoking the project hook. The hook remains optional. |
| Global ~/.codex configuration changed by this Goal | PASS | No global Codex configuration was written or copied. |

Manual action: open the WSL checkout ~/projects/RobotSim in the local Codex
client, inspect /hooks, verify that the RobotSim PreToolUse guard is listed,
and review its current definition. Trust it only if its behavior is acceptable;
otherwise leave it disabled. Do not bypass hook trust or approval controls.

## Local Environment Matrix

| Capability | Status | Observed state |
|---|---|---|
| WSL distribution | PASS | Ubuntu-22.04, Ubuntu 22.04.5 LTS, WSL version 2; running during the audit. |
| C/C++ compiler | PASS | GCC/G++ 11.4.0 are available in the WSL host. |
| CMake | NOT INSTALLED | Not available on the WSL host PATH. |
| Ninja | NOT INSTALLED | Not available on the WSL host PATH. |
| Python | PASS | Python 3.10.12. |
| Docker | PASS | Docker Desktop Engine 29.8.1; docker run --rm hello-world succeeded. |
| Docker GPU access | PASS | nvidia-smi succeeded in nvidia/cuda:12.6.3-base-ubuntu22.04 with the WSL-visible RTX 4060. |
| Project development image | NOT INSTALLED | robotsim:humble-dev is not present locally; Compose configuration parses. No image build was started. |
| Unity installation/version | PASS | Unity Editor 6000.3.25f1 is installed in WSL and its version command succeeded. |
| WSLg display endpoints | PASS | DISPLAY, WAYLAND_DISPLAY, X11 socket, and Wayland socket are present. This is endpoint availability, not a visual rendering result. |
| Unity rendered sensors/performance | DEFERRED | The PR #3 Unity test was a -nographics PlayMode test. Visual correctness, RGB/depth/LiDAR alignment, and performance have not been validated. |
| MuJoCo on the current host | NOT INSTALLED | Host Python cannot import mujoco; the project dev image is not built. PR #3 records a successful isolated MuJoCo 3.3.6 experiment, which is historical spike evidence, not a currently installed host package. |
| ROS 2 Humble | NOT INSTALLED | /opt/ros/humble/setup.bash and the host ros2 command are absent. The project dev image is also not built. |
| Unitree G1 vendor dependencies | NOT INSTALLED | third_party contains only LOCK.md; unitree_sdk2py and cyclonedds are unavailable to host Python. The G1 model/step test is recorded in PR #3, but its isolated source and package copies were temporary. |
| AgiBot X2 vendor dependencies | NOT INSTALLED | No X2 vendor source or runtime was found under the repository's third_party or /opt paths checked. Only the project-owned adapter placeholder and documentation are present. |

No additional ROS 2, MuJoCo, or vendor stack was installed or built merely to
make this matrix green, and the pre-existing Unity installation was not
changed. Host-side CMake/Ninja/ROS 2 absences are compatible with the
repository's documented Docker-first dependency boundary, but the project
container itself still needs its own acceptance run.

## Repository Hygiene

| Item | Classification | Finding/action |
|---|---|---|
| apps/unity/project/Assets, Packages, ProjectSettings | Durable project source | Tracked on spike/physics-ownership / PR #3, not yet on main. No source assets were moved or deleted. |
| Unity Library/ and Logs/ left in the local checkout | Generated/cache artifact | Unity Editor was not running. Removed these exact directories (about 2.0 GB and 7.3 MB); Unity regenerates them. |
| Unity Temp/, obj/, build output | Generated/cache artifact | Not present after the branch switch and cache audit. PR #3's .gitignore adds the Unity-generated directory rules. |
| /tmp/robotsim-mj-spike | Temporary experiment | Contained the isolated Unity project, MuJoCo 3.3.6 native/Python copies, and test logs from the completed spike. No process used it. Removed (about 2.0 GB). |
| /tmp/robotsim-physics-spike.uc22XH | Temporary experiment | Contained the temporary Unitree/MuJoCo upstream clones used by the completed spike. No process used it. Removed (about 410 MB). |
| Unity UserSettings/ | Editor-local, user-owned | Preserved. It is the only remaining untracked path in the checkout; it contains local editor preferences/layout state and is not staged. |
| Python __pycache__/ from this audit | Generated test artifact | Removed the two exact directories created by this Goal's test/compile run. |
| third_party/ source and binaries | Durable lock file only | No downloaded vendor source or compiled output is present. third_party/LOCK.md still has TODO entries. |
| Datasets, weights, scratch files | Unknown/user-owned unless tracked | No such untracked files were observed; data/ is tracked only through .gitkeep. |

The origin/main .gitignore does not yet contain Unity project cache rules;
those rules are part of PR #3. They were not duplicated in this branch. The local
UserSettings/ path is preserved because it is editor-local state; do not stage
it as part of this backlog.

## Architecture State

ADR-0005 remains **Proposed** on PR #3. This Goal does not accept it or start
architecture-dependent production code.

- Unity-hosted MuJoCo is technically viable: PR #3 records the official matched
  MuJoCo Unity plug-in/native 3.3.6 pair passing model compilation, mj_step,
  changing qpos, and Unity Transform synchronization in one headless PlayMode
  test.
- Standalone MuJoCo remains the proposed v1 direction, not an accepted decision.
- Unitree SDK2/DDS end-to-end execution is not validated.
- Unity visual correctness, sensor alignment, and sustained performance are
  not validated.
- No transport protocol is selected.
- Snapshot latency/jitter and physics-to-sensor alignment evidence remain
  incomplete.

## Backlog Order

| Draft | Status | Depends on | Environment | Risk |
|---|---|---|---|---|
| 01. Pin upstream versions and license provenance | READY | None | Either | Review-gated |
| 02. Accept the Docker development baseline | READY | None | Local | Routine |
| 03. Reproduce the G1 MuJoCo model/step baseline | WAITING_FOR_ENVIRONMENT | 01, 02 | Either | Routine |
| 04. Validate Unitree SDK2/DDS command-state round trip | WAITING_FOR_ENVIRONMENT | 01, 02, 03 | Local | Review-gated |
| 05. Complete ADR-0005 evidence and architecture review | WAITING_FOR_ENVIRONMENT | 03, 04, timing/sensor measurements | Local | Review-gated |
| 06. Define accepted clock/state/transport contract | WAITING_FOR_DECISION | 05; re-check Cloud Engineering Contract | Either | Review-gated |
| 07. Add minimal ROS 2 workspace and G1 adapter | WAITING_FOR_DECISION | 04, 06 | Either | Review-gated |
| 08. Add Unity timestamped snapshot frontend | WAITING_FOR_DECISION | 05, 06, 07 | Local | Review-gated |
| 09. Validate Unity RGB-D output and alignment | WAITING_FOR_DECISION | 06, 08 | Local | Review-gated |
| 10. Validate Unity LiDAR output and frame/timing | WAITING_FOR_DECISION | 06, 08 | Local | Review-gated |
| 11. Validate X2 backend parity against accepted interfaces | WAITING_FOR_DECISION | 01, 06, 07; X2 dependencies | Local | Review-gated |

Dependency sketch: 01 + 02 -> 03 -> 04 -> 05 -> 06; 04 + 06 -> 07;
05 + 06 + 07 -> 08 -> 09/10; and 01 + 06 + 07 -> 11.

## Issue Drafts

These are complete issue bodies for later creation. GitHub issue-write access
was unavailable: neither Windows nor WSL has gh, and no GitHub issue-write
connector is available in this session. No interactive credential setup was
attempted and no issues were created. The public issue tracker showed zero open
issues when checked; closed issues were not audited. Creation can be retried
from an authenticated GitHub client later.

### Draft 01: Pin G1 upstream versions and license provenance

**Status:** READY\
**Objective:** Replace the TODO third-party lock entries with exact, reviewed
upstream commits and record the provenance and license boundaries for the G1
baseline.\
**Why it exists:** scripts/fetch_third_party.sh currently clones floating
default branches, while third_party/LOCK.md has no pinned commits. Reproducible
G1 work and redistribution review need stable inputs.\
**Scope:** Pin unitree_sdk2, unitree_mujoco, and unitree_ros2; record the
MuJoCo 3.3.6 baseline, model/source provenance, and license references. Make the
fetch procedure reproduce those commits without committing vendor binaries.\
**Non-goals:** Selecting the production transport, changing the proposed
architecture, or redistributing vendor assets without permission.\
**Dependencies:** None. Re-read the Cloud Engineering Contract if it has landed
before changing any shared engineering policy.\
**Deliverables:** Updated third_party/LOCK.md; reproducible fetch/checkout
instructions; license/provenance notes with exact upstream URLs and SHAs.\
**Minimum validation:** Fetch into an empty temporary directory, assert exact
commit SHAs, inspect license files at those commits, and run git diff --check.\
**Execution environment:** Either.\
**Risk:** Review-gated (dependency/version and license provenance changes).\
**Completion evidence:** Reviewable diff, recorded commit hashes and license
references, and command output showing every fetched checkout is at its pin.

### Draft 02: Accept the RobotSim Docker development baseline

**Status:** READY\
**Objective:** Build and smoke-test the existing robotsim:humble-dev image as
the reproducible ROS 2 Humble/MuJoCo development environment.\
**Why it exists:** Docker and Docker GPU access work, and Compose parses, but the
project image has not been built on this machine. Host CMake, Ninja, ROS 2, and
MuJoCo are intentionally absent.\
**Scope:** Run the documented Compose build and container; verify ROS 2 Humble,
CMake, Ninja, MuJoCo 3.3.6, package imports, and GPU visibility where required;
record image digest and exact commands.\
**Non-goals:** Installing a duplicate toolchain in the WSL host, adding vendor
SDKs before their pins are reviewed, or starting robot-runtime implementation.\
**Dependencies:** None; Compose configuration already parses.\
**Deliverables:** Build/smoke-test record and any narrowly scoped fixes needed
for the existing Docker baseline.\
**Minimum validation:** docker compose ... build; run the documented ROS and
MuJoCo smoke commands; verify the expected tool versions and nvidia-smi in a
GPU-enabled container.\
**Execution environment:** Local.\
**Risk:** Routine; split any dependency or base-image changes into a
review-gated follow-up.\
**Completion evidence:** Successful build and command outputs, image digest, and
no untracked build artifacts in the repository.

### Draft 03: Reproduce the official G1 MuJoCo model and stepping baseline

**Status:** WAITING_FOR_ENVIRONMENT\
**Objective:** Turn the prior isolated G1 model-load/step experiment into a
repeatable headless check using the reviewed upstream pins.\
**Why it exists:** PR #3 records both official G1 MJCF variants compiling and
advancing with MuJoCo 3.3.6, but the temporary source and Python package copies
have been removed and the result is not yet reproduced from the project setup.\
**Scope:** Load the pinned G1 models, verify dimensions and named joints,
advance physics, apply a bounded actuator input, and assert finite state and
expected state change. Keep the run headless and record timestep/version.\
**Non-goals:** Stable walking, policy/controller quality, SDK2/DDS, Unity
integration, real-time guarantees, or Sim2Real claims.\
**Dependencies:** Drafts 01 and 02.\
**Deliverables:** A project-owned repeatable smoke test and concise result record;
vendor files remain unmodified under third_party/.\
**Minimum validation:** Both supported G1 model variants compile and complete a
fixed step count; state remains finite and the selected joint responds to the
test input.\
**Execution environment:** Either.\
**Risk:** Routine.\
**Completion evidence:** Test command, pinned source/library versions, pass
result, and recorded model dimensions/timestep.

### Draft 04: Validate the Unitree SDK2/DDS command-state round trip

**Status:** WAITING_FOR_ENVIRONMENT\
**Objective:** Prove a local G1 LowCmd to simulator to LowState round trip
using the reviewed SDK2 and simulator versions.\
**Why it exists:** The source topology and message paths are understood from
inspection, but the official SDK2/DDS process has not run in this WSL
environment. Shared mjData access is an identified upstream concurrency risk.\
**Scope:** Launch the simulator and SDK2 bridge on an isolated ROS domain, send a
bounded test command, verify corresponding state publication, inspect thread
ownership/synchronization, and record process startup/shutdown behavior.\
**Non-goals:** Real hardware, walking quality, ROS 2 application topics, Unity,
patching upstream files in place, or claiming deadline performance.\
**Dependencies:** Drafts 01, 02, and 03.\
**Deliverables:** Repeatable local test procedure, topic/message evidence, and a
reviewed finding on mjData concurrency with any RobotSim-owned mitigation
isolated from upstream files.\
**Minimum validation:** A test command is observed by the simulator and a
matching state message is received; shutdown leaves no orphaned process.\
**Execution environment:** Local.\
**Risk:** Review-gated (vendor bridge concurrency and middleware boundary).\
**Completion evidence:** Logs/topic evidence, exact commits, test results, and
documented race analysis.

### Draft 05: Complete ADR-0005 evidence and architecture review

**Status:** WAITING_FOR_ENVIRONMENT\
**Objective:** Gather the bounded runtime evidence needed to accept, revise, or
reject the proposed standalone-vs-Unity physics topology.\
**Why it exists:** Unity-hosted MuJoCo is viable with the matched 3.3.6 pair, but
the standalone SDK2/DDS path and comparative timing/sensor evidence are still
incomplete.\
**Scope:** Re-run the minimal matched Unity pair if needed; validate the G1
standalone bridge; measure candidate snapshot rate, latency/jitter, process
restart behavior, and representative Unity sensor-time alignment. State
measurement conditions and reviewable thresholds before each run. Update
ADR-0005 with evidence and alternatives, then request maintainer review.\
**Non-goals:** Implementing a production backend, selecting a transport without
measurements, deleting the Unity-hosted option, or accepting the ADR on behalf
of the user.\
**Dependencies:** Drafts 03 and 04; local Unity visual/sensor checks are also
required for any claim about rendered output.\
**Deliverables:** Updated proposed ADR with measured comparison, limitations,
and an explicit decision request.\
**Minimum validation:** Repeated timestamped measurements for representative
load; Unity sensor/state alignment inspection; no claim based only on -nographics
for visual quality.\
**Execution environment:** Local.\
**Risk:** Review-gated (architecture decision and incomplete GUI evidence).\
**Completion evidence:** Reproducible logs/data, updated ADR, and human review
record; ADR status changes only after explicit acceptance.

### Draft 06: Define the accepted simulation clock, state, and transport contract

**Status:** WAITING_FOR_DECISION\
**Objective:** Define the smallest versioned state/time boundary shared by the
MuJoCo backend, Unity frontend, and ROS 2 consumers after ADR-0005 is accepted.\
**Why it exists:** The ADR currently specifies no transport or production
schema. Timestamps, frames, joint ordering, and simulation truth must be explicit
before independently implemented components can interoperate.\
**Scope:** Specify simulation timestamp and sequence semantics, reset/pause
behavior, root/joint/link state representation, units, frames, quaternion order,
and truth-vs-estimate ownership. Select a transport only against measured
latency, failure, deployment, and sensor-alignment requirements. Follow the
Engineering Contract if it is merged before this task starts.\
**Non-goals:** Inventing a competing language/runtime/testing policy, building a
large universal robot abstraction, or implementing production transport code.\
**Dependencies:** Accepted ADR-0005 and the separate Cloud Engineering Contract
when merged; until then use only existing repository policy.\
**Deliverables:** Reviewed contract document and versioning/compatibility
criteria with concrete examples.\
**Minimum validation:** Check the contract against G1 state, X2 process
requirements, and the Unity sensor timing evidence; identify loss/restart and
stale-state behavior.\
**Execution environment:** Either.\
**Risk:** Review-gated (architecture and interface contract).\
**Completion evidence:** Explicit approval, examples accepted by backend/Unity/
ROS 2 owners, and no unresolved units/frame/order fields.

### Draft 07: Add the minimal ROS 2 workspace and G1 adapter

**Status:** WAITING_FOR_DECISION\
**Objective:** Expose the first G1 simulator through the accepted RobotSim ROS 2
state and command boundary.\
**Why it exists:** ros2_ws/ has no packages, and the project needs a small
robot-owned adapter between vendor-specific SDK/DDS and higher-level ROS 2
consumers.\
**Scope:** Create the minimum package/launch/test layout required by Draft 06;
keep vendor calls under robots/unitree_g1/; map joint names and timestamps;
keep ground truth distinct from estimated state.\
**Non-goals:** Generic multi-robot frameworks, Unity rendering, navigation,
perception, manipulation, real hardware, or vendor source edits.\
**Dependencies:** Draft 04, accepted Draft 06, and the Cloud Engineering
Contract if merged.\
**Deliverables:** Minimal ROS 2 package, G1 adapter/configuration, and an
executable integration test against the simulator.\
**Minimum validation:** Build with the accepted workspace instructions; publish
state with correct names/frames/timestamps and send a bounded command through
the adapter in simulation.\
**Execution environment:** Either.\
**Risk:** Review-gated (runtime interface and vendor boundary).\
**Completion evidence:** Build/test output, topic/interface example, and
demonstration that higher-level code does not import the vendor SDK directly.

### Draft 08: Add the Unity timestamped snapshot frontend

**Status:** WAITING_FOR_DECISION\
**Objective:** Render backend-owned robot state in Unity without making Unity a
second physics authority.\
**Why it exists:** The matched plug-in proves an in-process path is viable, but
the proposed v1 architecture expects Unity to consume timestamped snapshots.\
**Scope:** Implement the chosen contract/transport consumer, map root and named
joint/link state to the rig, interpolate or select by simulation timestamp, and
handle stale or reordered snapshots. Verify Unity Transform synchronization.\
**Non-goals:** Unity-side robot dynamics, production sensor implementation,
choosing a transport before Draft 06, or claiming performance from one frame.\
**Dependencies:** Accepted Draft 05, Draft 06, and a stable producer/interface
from Draft 07.\
**Deliverables:** Unity consumer, mapping configuration, automated PlayMode
test, and a visible local scene check.\
**Minimum validation:** Deterministic sample snapshots produce expected
Transforms; stale/reordered timestamps are handled; WSLg rendering is manually
inspected.\
**Execution environment:** Local.\
**Risk:** Review-gated (architecture-dependent GUI integration).\
**Completion evidence:** PlayMode results, screenshot/video or equivalent visual
record, and documented timing limits.

### Draft 09: Validate Unity RGB-D output and time alignment

**Status:** WAITING_FOR_DECISION\
**Objective:** Produce timestamped RGB and depth observations from the Unity
scene that align with the accepted robot/world state.\
**Why it exists:** RGB-D perception is an initial platform target, but no
rendered or aligned sensor output has been validated.\
**Scope:** Configure one camera, publish image/depth/camera metadata with frame
and acquisition time, and validate projection/depth consistency against a
known scene target and backend timestamp.\
**Non-goals:** Perception models, grasping, sensor noise research, or claims of
photorealism/real-sensor parity.\
**Dependencies:** Drafts 06 and 08; re-check the Engineering Contract if it has
landed before implementation.\
**Deliverables:** Reproducible scene/configuration, sensor output example, and
alignment test.\
**Minimum validation:** Check image dimensions/encoding, depth units, intrinsics,
frame IDs, timestamps, and projected target position/depth within stated
tolerances.\
**Execution environment:** Local.\
**Risk:** Review-gated (sensor interface and local rendering).\
**Completion evidence:** Captured RGB/depth sample, automated checks, visible
scene verification, and stated limitations.

### Draft 10: Validate Unity LiDAR output, frames, and timing

**Status:** WAITING_FOR_DECISION\
**Objective:** Establish a timestamped Unity LiDAR/point-cloud baseline against
the same world and robot state used by the backend.\
**Why it exists:** LiDAR is an initial platform target, but its geometry,
coordinate convention, acquisition cadence, and state alignment have not been
measured.\
**Scope:** Implement one sensor configuration; emit a scan/point cloud with
explicit frame, units, acquisition time, and rate; compare measurements against
known geometric primitives.\
**Non-goals:** SLAM, localization, navigation, multi-sensor fusion, or real
hardware parity.\
**Dependencies:** Drafts 06 and 08; re-check the Engineering Contract if it has
landed before implementation.\
**Deliverables:** Reproducible sensor configuration and a repeatable scan/frame/
timing validation.\
**Minimum validation:** Confirm expected ranges/angles, frame transforms,
timestamp relation to the backend snapshot, and output rate under the intended
Unity rendering mode.\
**Execution environment:** Local.\
**Risk:** Review-gated (sensor interface and local rendering).\
**Completion evidence:** Captured point cloud/scan, automated geometry/timing
checks, and a visible local verification record.

### Draft 11: Validate AgiBot X2 against the accepted backend interfaces

**Status:** WAITING_FOR_DECISION\
**Objective:** Confirm that X2's official sim_mujoco/MC process topology can
fit the accepted RobotSim state/command contract without leaking vendor logic
into shared packages.\
**Why it exists:** X2 is the planned second backend, and upstream documentation
describes a process-oriented simulation path, but no X2 model or runtime is
available in this checkout.\
**Scope:** Review upstream source/license and dependency requirements; load the
permitted model, run a bounded simulation smoke test, and map the minimum state
and command boundary through an X2-owned adapter.\
**Non-goals:** Committing restricted vendor packages/assets, implementing
manipulation, changing the shared contract solely for vendor convenience, or
testing physical hardware.\
**Dependencies:** Draft 01, accepted Draft 06, Draft 07, and authorized access to
the required X2 packages/models.\
**Deliverables:** X2 dependency/provenance record, adapter/configuration, and a
repeatable model/runtime interface test.\
**Minimum validation:** Model compiles and advances; state/command names, frames,
units, and timestamps map to the accepted contract; any unsupported field is
explicitly recorded.\
**Execution environment:** Local.\
**Risk:** Review-gated (vendor dependency, licensing, and second-backend
compatibility).\
**Completion evidence:** Exact upstream references, license review, smoke-test
output, and a G1/X2 interface comparison.

Localization/mapping, manipulation, physical Sim2Real, and broad CI expansion
remain outside this first issue tranche. Revisit them after the clock/state
contract and at least one end-to-end sensor/backend path are accepted. Do not
start any READY draft automatically as part of this Goal.
