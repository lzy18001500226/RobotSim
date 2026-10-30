# Technical note: AgiBot X2

Expected components:
- official X2 URDF / MJCF
- AimDK
- MC
- sim_mujoco
- X2 grasp demo as a manipulation reference

X2's official simulation stack is useful because MC can provide vendor locomotion capabilities. E1R / Gemini335 simulation remains a separate project layer.

## Pinned community navigation reference

`reference/AgiBot-Community/agibot-x2-navigation` is a read-only checkout of
https://github.com/AgiBot-Community/agibot-x2-navigation at immutable commit
`9ad1ad8af1b47b1d8fda42c3ca9c828bc5625629` (repository `LICENSE`: MIT).
It is a reference for FAST-LIO, NDT-OMP localization, OctoMap, Nav2 and a
`/cmd_vel` to X2 velocity bridge. It is not an x86 PNC binary release and must
not be run or copied into the runtime directly: its physical-robot source name,
minimum velocities, map/TF assumptions and dependencies require a bounded
adapter plus separate x86 MuJoCo validation.

## Issue #80 RAICOM scene namespace contract

The installed 2026-05-22 MC/simulator model manager uses the native X2
`model_info/default.yaml` names. Its `odom` entry is the body `pelvis`; its
active-joint lists contain 31 unprefixed joints. The native MJCF also contains
`floating_base_joint`, 31 `motor_<joint>` actuators, and 67 sensors. Scene
composition must preserve these names. A robot `<attach prefix="x2_">`
renames all of them and can compile successfully while breaking the runtime.

Local01 source commit `95d76d82ea7def0dd719da4ebe07f9654044415e` remains
immutable. The research scene adapter generates a wrapper with exactly one
namespace delta: X2's attach prefix becomes `""`. Room `raicom_` and mug `mug_`
prefixes, model paths, object placements and physical properties are checked
separately. No vendor XML or proprietary binary is patched. Compilation with
an empty robot prefix has no same-namespace name collisions.

Run the offline audit from the existing Issue #80 worktree:

```bash
/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  scripts/research/issue80_local01_scene_route.py \
  --poses "$LOCAL01_WORKTREE/simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json" \
  --runtime-root "$ROBOTSIM_ROOT/simulation/mujoco/raicom2026" \
  --output-parent "$ROBOTSIM_ROOT/reviews/issue-80"
```

`name_audit.json` contains every original/corrected body, joint, actuator and
sensor name, missing-name inventories, compiled-array equivalence and static
FK evidence. The generated container wrapper is compiled again with its
declared mount paths resolved on the host. The corridor check uses native body
ancestry rather than the retired `x2_` name heuristic. None of these checks
calls `mj_step` or validates live MC behavior. The prototype is `KEEP-AS-TOOL`
for bounded Issue #80 research, not production navigation.

Known limitations must remain explicit:

- Five `body-*-imu_1` entries in the YAML are absent from the untouched native
  MJCF too. They are a baseline discrepancy; do not fabricate replacement
  sensors or claim every configured secondary IMU lookup succeeds.
- The existing combined scene uses MuJoCo's 0.002 s timestep while native X2's
  scene uses 0.001 s. The naming correction preserves that existing combined
  scene value; it does not establish complete timing compatibility.
- Saved successful native Reset telemetry reports z around 0.67956 m. The YAML
  odom source is `pelvis`; `body-linear-pos` is a separate `imu_0` framepos
  sensor. This supports height-bearing telemetry rather than a conventionally
  zero planar z, but proprietary publisher code is unavailable. The translated,
  rotated room's Reset anchor was subsequently observed in scene-world XY/yaw
  (2026-10-26 capture below); this is not an independent live ground-truth stream.
  Never replace missing-body zero telemetry with an assumed 0.68 m reading.
- The established `SIMULATION_ONLY` FoundationWalk overlay has left/right knee
  action scales 0.19 (vendor 0.25) and `min_run_velx=0.1` (vendor 0.5).
  Source SHA-256 is `b3081e9bca2035d4a986657bdc03b79ca0ed5d4e852c7972d86816f569acc2e9`;
  overlay SHA-256 is `6eb712872248c1c377b27c1bb58fdd5b44bdd4262e0271e779d0040ee82964e2`.
  Namespace repair changes neither. Results with that overlay are not
  unmodified-vendor-configuration acceptance.

The dedicated runner is
`scripts/research/issue80_raicom_reset_only.py`; its companion
`issue80_raicom_reset_only_observer.py` subscribes to state topics only. It
starts the existing bridge-isolated simulator/MC environment with the runtime
tree mounted read-only, checks the pinned scene and native names, and permits
one verified native Simulation/Reset click. Its 30 s wall cap is backed by a
26 s container-stop watchdog. It has no route, MC action service client, source
registration client, or velocity publisher. A Reset-only result is not a
standing or locomotion qualification.

The single authorized attempt `50ef59e2-f262-4ef8-a964-406ae122df1e` loaded
the corrected RAICOM scene wrapper and passed runtime config/name checks, but
the ROS observer failed during construction because it assigned to rclpy's
read-only `Node.subscriptions` attribute. The container watchdog stopped the
run before readiness; `native_reset_dispatch_count=0`, no Reset event exists,
and no odometry/IMU/joint trace was captured. Container absence was verified.
The observer startup error path is now recorded and fail-fast in code, but this
attempt's live authorization is spent. Any further simulator/Reset run needs a
new explicit authorization. Do not use the first-motion route CLI for a
Reset-only task.

A verified button click and upright odometry alone cannot yield a Reset-only
PASS. The next authorized capture must observe either a post-Reset odometry
generation/timestamp rebase or a measured pelvis-pose discontinuity from the
fresh pre-Reset snapshot; otherwise it reports `RESET_EFFECT_UNCONFIRMED`.

The same Humble/rclpy image now has a construction-smoke mode on the observer.
It creates the five state subscriptions, then requires zero publishers and
zero service clients before teardown. In this image rclpy creates a
`/parameter_events` publisher even with parameter services disabled, so the
observer explicitly destroys that unused framework publisher. The construction
smoke and its injected-failure path must run before a future Reset-only episode.

The renewed Reset-only attempt `8bf0e747-10bc-4569-97d4-ea5fdb3db55f` reached
the corrected scene and logged `Odom initialized: body_id=1, name=pelvis`, but
MC aborted before telemetry readiness because its startup copies
`mc_sim.yaml.dump` beside the readonly vendor runtime config. No Reset was
dispatched. The runner now stages that one generated dump file in its evidence
directory and mounts only that exact file writable beneath the otherwise
readonly runtime tree. That correction alone was insufficient: the subsequent
authorized capture required generated `sim/*.yaml` files as well.

### Read-only runtime generated-config prelaunch

Run the small no-physics prelaunch before an independently authorized episode:

```bash
/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  scripts/research/issue80_raicom_prelaunch.py --host
```

It uses the pinned image, no network, readonly runtime, and state-only observer
construction. It tests dump truncation and exact generated-config write paths;
the readonly parent and cross-mount rename failures remain explicit. The
Reset-only runner additionally provides an 8 MiB tmpfs at the exact native
`mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim` directory, seeds the three existing
YAML files, and preserves MC's generated files and symlink descriptions before
cleanup. The binary, source configurations, ONNX and vendor model stay readonly.
This option defaults off for other runners. Generated output is not a new MC
policy. Its hashes must be compared with the staged source seed.

The 2026-10-26 capture `2e141d98-6954-4d9d-ab48-a465d49e274e` demonstrated
complete MC startup, five fresh streams and one effective native Reset. All
three generated YAML files were byte-identical to their seed. Reset restored
raw HAL odom to `(-1.5,-1.5,0.6796076,pi/2)`, agreeing with scene-world START,
not the Local01 planned Reset-local `(0,0,0.68,0)` anchor. Odom source stamps
continued increasing: physical discontinuity, not timestamp rebasing, proved
the Reset effect. The first fresh upright sample must be used for that anchor
comparison; a later fallen sample cannot establish the Reset frame. Final
posture is reported independently and must still fail closed.

The robot remained `PASSIVE_DEFAULT` in all 32 CommonState samples. All 194
leg-command messages had zero stiffness, damping and effort. Upright samples
spanned only 0.195602 s before pitch reached -0.283761 rad. The capture therefore
remains BLOCKED for sustained upright readiness. No SD/LD, input registration,
velocity publication or navigation occurred. Never spend another Reset-only
episode expecting passive physics alone to produce standing control. A future
Reset-to-SD takeover experiment requires separate authorization, fresh posture
at the service request, unchanged guards and complete raw response capture.

The source Local01 SE(2) transform is an offline planned-local frame; do not
apply its local waypoints directly to vendor raw scene-world odometry. Preserve
both frames explicitly. The existing 0.002 s room timestep versus 0.001 s native
scene also remains unqualified for active MC motion; telemetry frequency alone
does not demonstrate integration-time compatibility.

### Static table-route planning tool

`issue80_raicom_table_plan.py` reuses the pinned scene wrapper and route schema,
uses only `mj_forward`, and writes a unique governed directory with JSON and an
SVG. It checks source-zero and runtime-nominal postures, active environment and
self contacts, source limits, 50 mm XY uncertainty and sampled yaw uncertainty.
The remaining 20 mm clearance gate is a planning margin, not a changed runtime
safety threshold. Two static postures do not bound the dynamic gait envelope.
Its `table_route_raw_world_ODOM_OFFLINE_ONLY.json` uses scene-world numeric
coordinates for the observed raw HAL `odom` channel. Planned-local coordinates
remain separately named report fields, never a second route config mislabeled
as raw odom. This repairs the offline artifact contract only; the live short
route's source transform and fail-closed Reset-anchor gates remain unchanged.

The source docking `(1,-1.5,0.68,0)` has only 11.3 mm conservative clearance
under nominal posture and fails that additional margin. The separately labeled
`(0.95,-1.5,0.68,0)` review candidate has 61.3 mm; it does not replace Local01's
source target or establish manipulation IK. The initial -90 degree alignment
is geometric only: stationary turning is unqualified, and the 1.6 m radius
moving arc at the command caps does not join the straight corridor. Consequently
the full-distance route is OFFLINE_ONLY. Its approximately 2.5 m travel needs
at least 20.8 s even at commanded speed; measured historical progress suggests
about 72.5 s / 15 transit windows, plus independent stop/rearm and alignment.
Neither the old 15 s motion budget nor this estimate authorizes a long run.

### Bounded Reset-to-SD active standing tool

`issue80_raicom_stand_takeover.py` reuses the Reset-only launch, verified native
Reset, generated-config isolation, scene/name checks and cleanup. Its separate
observer permits only SetMcAction(STAND_DEFAULT) once and read-only GetMcAction;
it has five state subscriptions and no publisher or input-source client. Run
`issue80_raicom_prelaunch.py --host --standing` to construct these exact rclpy
entities without simulator/MC physics. This does not authorize a live episode.
Disposition: KEEP-AS-TOOL for Issue #80 isolated research; not a production
startup policy or permission to execute new physical episodes.

The standing client is discovered before Reset. Three fresh advancing upright
odom samples and observed Reset effect gate the sole SD request; request-time
pose age is capped at 50 ms. Existing 0.52 m height, 15 degree tilt, source leg
position/velocity limits, telemetry freshness and 0.20 m/s planar speed guards
remain. An active hold requires fresh agreeing Get/Common SD/status=100,
body=1, pinned-build FSM=4, valid nonzero standing gains/effort, and at least
2.5 seconds/25 unique observations. Every broken qualification resets that
hold; a raw FSM=6 under SD remains fail-closed, never a standing PASS. The host
26 s Timer watchdog is independent of the observer, but not an OS guarantee
against parent-process SIGKILL; container absence still requires verification.

One authorized capture at `faa6848`, attempt
`5d005147-f1e7-4e56-8d7d-9e4ef3e51e11`, sent SD 16.800 ms after first fresh
upright pose (pose age 2.465 ms). The service returned header.code=0,
CommonState.SUCCESS=1 in 12.464 ms. MC's own log confirms its shortest route
PASSIVE_DEFAULT -> JOINT_DEFAULT -> STAND_DEFAULT without an external JD call.
The standing runner's PreRunOnce was logged, and gains reached stiffness=150,
damping=5. Feedforward effort and HAL joint-state effort stayed zero; these
fields do not independently measure simulator PD actuator torque.

The run stopped after only 59.779 ms of post-request odom observation:
CommonState reported SD/status=100, FSM=6, body=0 while the latest Get response
was JD/status=200. Last recorded pitch was -0.015375 rad and height 0.674466 m;
no post-Reset physical fall was recorded before the guard. Qualification is
BLOCKED, not an SD rejection, late dispatch, or demonstrated model instability.
Raw 6 already existed in PASSIVE, and no explicit new protection cause was
logged. A stale/category-update conflict versus a genuine protection condition
remains unresolved. The guard is retained; its later diagnostic label is
`SD_READBACK_CONFLICT_RAW_FSM6`, without changing the original run artifacts or
the stop decision. No 2.5 s active hold, locomotion, or automatic retry occurred.

The saved successful Run65 differs: it sent JD explicitly, waited approximately
0.244 s and observed body=1/FSM=1 before SD. Its CommonState still showed JD
with raw FSM=6 earlier in that successful initialization, then SD with FSM=1
at +97 ms and FSM=4 at +187 ms after SD. Thus target-mode acceptance and FSM
category readiness demonstrably need separate gates. This comparison supports
reviewing explicit native JD readiness staging, not ignoring raw 6 or asserting
the present 60 ms capture proves that SD can never stand. Run65's floor scene
and timestep differ too, so the sequence difference is not isolated causation.

The scene remained at 0.002 s and native baseline at 0.001 s. MC command and
sim IMU configs specify 500 Hz, joint-state publication 1000 Hz. Observed ROS
delivery intervals and missed-deadline warnings are retained; they neither
prove a complete simulation-time compatibility contract nor explain the raw
FSM conflict on their own. A future separately authorized diagnostic needs an
explicit transition/readback contract before any relaxation of that conflict
gate; do not infer safety merely from a successful service response.

### Explicit native JD-readiness protocol

The same dedicated standing tool accepts `--protocol jd-sd`. It requests JD
once only after confirmed Reset effect and fresh safe advancing posture;
requires accepted JD plus fresh agreeing Get/Common JD/status100, FSM1/body1,
valid active gains and source joint safety; then requests SD once. Readiness
is observation-driven, not a 244 ms sleep. JD readiness is bounded to 1 s;
SD readiness to 2 s; a qualified active hold requires 2.5 s and 25 unique
observations. Post-Reset capture is bounded to 7 s, within the unchanged
15 s observer, 26 s container watchdog and 30 s runtime cap.

Run65 supports unchanged initial raw6 during the JD preparation phase only.
It remains TRANSITION_PENDING and never grants SD permission. After a normal
FSM has been observed, raw6 returning is protection and stops the episode.
After verified JD/FSM1/body1, SD may have the documented asynchronous JD/SD
readbacks while FSM1 changes to FSM4; mismatches do not advance the hold.
Unknown categories/body contracts and protection ambiguity fail closed.
Duplicate source stamps do not refresh state or tick readiness; regressions
stop. The original direct-SD protocol remains separate.

One authorized JD-SD episode, `bbe4b6af-b5db-411a-ac42-835cbb144579` at
`61535ca`, was BLOCKED before observer launch or Reset: inherited
`ros2 topic info --verbose` exceeded its 3 s subprocess deadline. Both MC and
MuJoCo module-ready logs were saved; runtime was 16.838 s with verified
container removal. No Reset/JD/SD request or standing telemetry was obtained.
This is an infrastructure pre-Reset result, not a failed JD/SD qualification.

For this protocol only, subsequent code moves the same zero-velocity-publisher
check into the already initialized rclpy observer. Missing graph proof, any
publisher, stale telemetry, or a failed client gate blocks Reset. The graph is
rechecked before actions and periodically; no CLI subprocess or new publisher
is created. The parent validates that proof before dispatch. Default Reset-only
and direct-SD paths keep their existing inspection. Real Humble no-physics
smoke verifies the API and five subscriptions/zero publishers/two clients;
it does not validate the repaired path against live MC. No retry was executed.

Execution step logs now retain monotonic start/end values, since reconstructed
wall-time fields alone cannot establish elapsed ordering. A blocked result
before observer launch can positively report zero action requests; a missing
summary after launch remains UNKNOWN instead of inventing zero. Both semantics
have offline regressions. Any next JD-SD physics episode needs new authorization.

The replacement JD-SD episode `6e6b4b60-47de-4c62-9f3f-5a6e1c0fb7ef` verified
the repaired rclpy graph path in the live isolated container: zero velocity
publishers at READY and again immediately before JD/SD actions. Reset, one JD
and one SD were dispatched. JD completed its observed gate in 300.726 ms;
SD was accepted 6.395 ms after its request. The robot then exceeded the
unchanged 0.20 m/s planar-speed safety bound (0.209026 m/s) 41.990 ms after
the SD request, before a new CommonState could establish SD readiness. z,
roll/pitch and source joint position/velocity limits were still valid; this is
not a measured fall, but it is a real physical safety-gate failure. The
qualified active hold is 0.0 s, so the result is BLOCKED and no locomotion may
follow. `posture_safety_reason` now records this future condition as
`PLANAR_SPEED_LIMIT` without changing the threshold. Do not retry under that
authorization or attribute the speed rise to timing warnings without further
evidence.

For a separately authorized future attempt, the command shape is:

```bash
DISPLAY=:0 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  /home/lzy18001500226/robotsim-issue80-x2-navigation-20261019/scripts/research/issue80_raicom_reset_only.py \
  --attempt-id <fresh-uuid> \
  --local01-scene-world-poses /home/lzy18001500226/robotsim-issue77-raicom-x2-reuse/simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json \
  --evidence-parent /home/lzy18001500226/projects/RobotSim/reviews/issue-80
```
