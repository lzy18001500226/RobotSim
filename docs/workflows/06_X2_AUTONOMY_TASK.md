# X2 autonomous fetch task workflow

This workflow defines the handoff contracts for the RAICOM-style task. It does
not authorize physics stepping, mechanical-model changes, or hardware motion.

## Frozen inputs

- Robot/gripper selection: `X2-OmniPicker-STL` through the registry loader.
- World source: one scene adapter for the selected map, with explicit map,
  LiDAR, RGB-D, table, interaction-zone, and delivery-zone frames.
- Ground truth: allowed only for offline evaluation and estimator comparison;
  never as a navigation, docking, or grasp-control input.

## Task state machine

```text
RESET -> STAND -> NAVIGATE_TO_INTERACTION -> RECEIVE_REQUEST
      -> NAVIGATE_TO_TABLE -> DOCK_FOR_PERCEPTION -> ESTIMATE_OBJECT
      -> REFINE_BASE_STANCE -> PLAN_ARM -> PREGRASP -> CONTACT_GRASP
      -> NAVIGATE_TO_DELIVERY -> PLACE -> REPORT_COMPLETE
```

Advance a stage only on its measured acceptance evidence. A rejected candidate
keeps that stage unpassed; it does not automatically terminate the mission.
The task executor owns recovery under [engineering progress and recovery](goal_driven_development.md#engineering-progress-and-recovery):

- Reacquire stale perception/localization, refine the stance, or replan an
  unreachable approach within the existing permissions and trial budget.
- During carry/placement, verify that a qualified hold remains valid before
  replanning a landing pose. If safe holding cannot be maintained, stop the
  affected execution; a failed plan is not permission to release the object.
- Treat docking and placement targets as feasible regions defined by task
  geometry, reachability, contact and support, rather than one exact waypoint.
  A planner-only angle cutoff is not itself a physical joint limit.
- Preserve real stability, slip, collision, source-limit and numerical checks.
  Recoverable planning must not advance through an actual physical failure.

Validate generalization through the same pipeline at intermediate or sampled
positions inside the accepted region. Separate per-position successes are useful
evidence, but do not establish continuous-region or end-to-end success.

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

## Workstream split

1. **LiDAR navigation and randomized obstacles** consumes only map/LiDAR/odom
   and produces a stable docking result. Obstacles are generated from a seed
   recorded with each run; their truth pose is not available to the planner.
2. **RGB-D docking and manipulation** consumes the navigation stop result and
   estimated object pose, then chooses a reachable stance and arm plan. It
   cannot read object truth after the estimator boundary.
3. **Task execution** is the sole owner of state transitions and retries.

The workstreams may investigate independently, but maintained code integrates
sequentially into one active branch and one task PR. Their only shared model
input is the frozen registry identity above.
