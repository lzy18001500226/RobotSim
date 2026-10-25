# Technical note: AgiBot X2

Expected components:
- official X2 URDF / MJCF
- AimDK
- MC
- sim_mujoco
- X2 grasp demo as a manipulation reference

X2's official simulation stack is useful because MC can provide vendor locomotion capabilities. E1R / Gemini335 simulation remains a separate project layer.

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
  rotated room's live world-versus-local XY/yaw convention remains unverified.
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

For a separately authorized future attempt, the command shape is:

```bash
DISPLAY=:0 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  /home/lzy18001500226/robotsim-issue80-x2-navigation-20261019/scripts/research/issue80_raicom_reset_only.py \
  --attempt-id <fresh-uuid> \
  --local01-scene-world-poses /home/lzy18001500226/robotsim-issue77-raicom-x2-reuse/simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json \
  --evidence-parent /home/lzy18001500226/projects/RobotSim/reviews/issue-80
```
