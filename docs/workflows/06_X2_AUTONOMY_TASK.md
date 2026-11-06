# X2 autonomous fetch task workflow

This workflow defines the handoff contracts for the RAICOM-style task. It does
not authorize physics stepping, mechanical-model changes, or hardware motion.

## Frozen inputs

- Robot/gripper selection: `X2-OmniPicker-STL` through the registry loader.
- World source: one scene adapter for the selected map, with explicit map,
  LiDAR, RGB-D, table, interaction-zone, and delivery-zone frames.
- Ground truth: permitted in explicitly labeled SIMULATION_ONLY debugging and
  the maintainer-authorized Stage 2 truth-first grasp/stance validation. Record
  the target source. Truth may not silently cross an estimator acceptance
  boundary or be reported as LiDAR/depth-camera navigation or grasp success.

## Current milestone and observed status (2026-11-06)

Current Stage 2 scope is fixed table stance -> approved lower concave-pad
support-circle IK -> actual opening/approach/arrival -> closure -> >=50 mm
object rise -> >=1 s unsupported hold. Use truth first to isolate execution;
fresh RGB-D substitution remains separately required for perception acceptance.
Stage 3 adds perception-driven base repositioning; do not expand into it before
the fixed-stance grasp is qualified.

Observed local standing/docking and static IK passes do not close Stage 2.
The latest opening-first native episode failed actual full opening before
walking. Earlier approach evidence also exposed a frozen/stale-base planning
error. Grasp and lift remain unqualified. See the [measured findings](../research/issue46_truth_grasp_execution_findings.md)
for exact trace identities; do not promote a prior model's PASS to this setup.

Use the [scoped parameter authorization](../engineering/model_integrity.md#c1-standing-authorization-issue-46-omnipicker-simulation-parameter-debugging)
for existing gripper gain/joint-resistance diagnostics; ordinary permitted
parameter debugging does not require another approval. Model identity,
geometry, topology, contact and force-limit changes retain their review gates.

## Task state machine

```text
RESET -> STAND -> NAVIGATE_TO_INTERACTION -> RECEIVE_REQUEST
      -> NAVIGATE_TO_TABLE -> DOCK_FOR_PERCEPTION -> ESTIMATE_OBJECT
      -> REFINE_BASE_STANCE -> PLAN_ARM -> PREGRASP -> CONTACT_GRASP
      -> NAVIGATE_TO_DELIVERY -> PLACE -> REPORT_COMPLETE
```

This is the intended end-to-end lifecycle, not a statement that all stages run
successfully today. The current grasp implementation has no qualified lift
executor or delivery result. Every transition is fail-closed. A failed localization, stale frame transform,
unreachable grasp, unsupported object, contact failure, or stability failure
returns a reasoned failure state rather than moving to the next stage.

## Interface contracts

| Producer | Required output | Consumer |
| --- | --- | --- |
| Localization/navigation | timestamped `map -> base_link`, covariance, velocity, route/stop result | docking and task executor |
| RGB-D perception | timestamped `camera_optical -> object`, mask/point support, confidence | docking and grasp planner |
| Docking | candidate `base_link` stance poses, clearance and reachability status | navigation and IK |
| Manipulation/IK | joint-space pregrasp/approach plan plus collision checks | control adapter |
| Grasp validation | bilateral contact, table-support transfer, object COM motion, safety result | task executor |

All spatial contracts use REP-103 units and frames. Map-specific semantic
locations are configuration data, never hard-coded in planning code.

## Standing OmniPicker grasp sequence

For the frozen `X2-OmniPicker-STL` model, table grasp development follows one
continuous sequence.  Do not treat a static pose, a working gripper slider, a
passing arm IK solve, or a completed short physics probe as a completed grasp.

1. Keep the accepted robot, both STL hands, source meshes, collision coverage,
   table, and object geometry fixed.  Use a vertically supported object on
   the table.
2. Search the robot base pose, base yaw, object position on the supported
   table area, knee bend, waist yaw/pitch, and wrist/arm pose.  The ordinary
   posture constraint is waist roll within 2 degrees of the stable standing
   state.  Prefer straight posture, then knee bend or forward waist pitch;
   reject a side-bent waist as a grasp workaround.
3. Screen every candidate with `mj_forward` before stepping: complete hand
   pose, bilateral distal-pad contact at the requested close command, no
   non-pad object intersection, no robot-table collision, finite source joint
   margins, and object table support.
4. Start physics from a stable table-preplaced standing state.  First hold the
   stance, then ramp the upper body to pregrasp.  Record actual hand-to-object
   error; a static target is not sufficient.
5. When selecting the opening-first startup path, command open during
   initialization and retain it through walking under exclusive hand ownership.
   Verify actual opening before walking and again before approach; command
   publication or a fixed sleep is not opening confirmation. Preserve an
   already-open hand at the docking handoff rather than closing it to reactivate.
   This desired startup path has not passed physical opening qualification.
6. Only after the physical pregrasp remains stable and the actual hand is in
   the screened contact region may the gripper close.  Require sustained
   bilateral distal-pad contact before attempting lift.
7. Lift in order: a small clearance probe, then a 50 mm lift and hold.  A
   grasp PASS requires table-support transfer, opposing pad contact, object
   COM rise of at least 50 mm, and stable hold.  No weld, object qpos write,
   hidden support, or collision exclusion may substitute for those checks.

Failure handling remains local to the failed stage:

- Static-screen failure: adjust only base stance, supported object placement,
  knee/forward-pitch posture, or arm/wrist approach pose.
- Standing or pregrasp failure: correct the initial state or the smooth
  transition before issuing a close command.
- Hand-to-object error or missing bilateral contact: search the screened
  stance/IK neighbourhood using the measured MuJoCo hand pose; do not merely
  increase gripper force.
- Contact or lift failure: inspect contact forces, table support, root motion,
  and object COM before trying the next bounded candidate.

Each failed candidate must preserve its result and select the next local
candidate.  The workstream stops for human review only at a visual posture
choice, a source/model change, a true physical grasp result, or when the
remaining failure is outside the documented controls and evidence. Apply the
explicit parameter authorization above before treating ordinary simulation
control tuning as an unapproved mechanical change.

## Workstream split

1. **LiDAR navigation and randomized obstacles** consumes only map/LiDAR/odom
   and produces a stable docking result. Obstacles are generated from a seed
   recorded with each run; their truth pose is not available to the planner.
2. **RGB-D docking and manipulation** consumes the navigation stop result and
   estimated object pose, then chooses a reachable stance and arm plan. It
   cannot read object truth after the estimator boundary in perception-qualified
   operation. The labeled Stage 2 truth diagnostic is a separate mode; it must
   never claim fresh RGB-D acceptance.
3. **Task execution** is the sole owner of state transitions and retries.

The workstreams may investigate independently, but maintained code integrates
sequentially into one active branch and one task PR. Their only shared model
input is the frozen registry identity above.
