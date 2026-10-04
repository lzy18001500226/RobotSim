# Issue #43: Physics and acceptance audit

## Scope and evidence boundary

This report audits `NVlabs/humanoidmimicgen` at `d82844dcec242c82d6b82628ccc45933c3ad5cbd` against the reported Local02 run of G1 `LMDrillPnP90` through WBC-goal playback (`final_success=true`). The source was inspected at that exact commit. This is a source-path audit; the Local02 run result is supplied evidence and was not independently rerun here. The RobotSim branch carrying this report is based on main `11aebeb60a1d3c1c2f580c8f4d6bd6907581fd33`; it contains documentation only.

The audited path is HDF5 source-demo WBC-goal playback. The separate `state` playback mode restores a stored simulator state on every frame and explicitly does not validate action or physics fidelity. Do not use a `state` playback success as physical evidence.

## What `final_success=true` establishes

For each HDF5 episode, playback calls `env.reset(seed)`, then calls `env.reset_to` once with `states[0]` (and, when present, the episode model XML). In WBC-goal mode each subsequent recorded WBC goal is passed to `ActionEnv.step`: the controller observes the current state, sets the goal, calculates a joint action, and advances the environment. There is no per-frame state restore in this loop. After every active step the code reads the resulting flattened MuJoCo state and compares it with the demonstration state; `--allow-state-divergence` makes that state-fidelity check non-gating.

`final_success` is the task-success bit from the last checked active-playback state. It means the final simulated state passed the task predicate below. It does not mean every stage succeeded, that the object was grasped, or that its path was physically continuous. The CLI's `--require-task-success` gate checks whether each episode had *any* successful step; it does not require the final step to be successful. These are separate from state-fidelity success.

## Physical object, state writes, and contact

`LMDrillPnP90` creates the drill from the `powerdrill_b01` MJCF as a non-static object. `MJCFObject` injects a free joint with damping, giving the object a seven-value free-joint `qpos` (position and quaternion) and six-value `qvel`. The model has visual mesh geoms with collision disabled and separate box/capsule collision geoms. For this task, the `ObjectConfig` defaults resolve to density `10`, friction `(1, 1, 1)`, `solimp=(0.998, 0.998, 0.001)`, and `solref=(0.001, 1)`; `MJCFObject` applies these to the geoms. Those simulation parameters affect the contact response, so the run record should preserve them.

There are two expected initializations before the active rollout: scene reset samples and writes the object's free-joint pose; then playback restores the *whole* recorded `states[0]` snapshot through `reset_to`. The latter does directly restore the object qpos once after `env.reset`, before the first WBC action. During the WBC-goal action loop, the audited code path sends robot joint actions to `env.step`; it has no object-qpos assignment or state restore. This supports the conclusion that carry motion in this playback path is advanced by MuJoCo stepping, not a per-frame object teleport. An independent runtime trace is still required to prove no external wrapper, callback, or patched environment performs an uninspected write.

The drill asset contains no weld or equality attachment. The audited task construction and WBC playback path do not add one. Thus the object is free and can move through ordinary contact/friction and constraint solving. That establishes that physical carry is possible; the `final_success` bit alone does not demonstrate that contact caused the specific observed motion.

The G1 model uses its three-finger hands. Each hand asset names collision geoms for thumb, middle, and index fingers, and the object exposes collision geom IDs. At runtime, MuJoCo provides contact pairs (`data.ncon`, `data.contact[i].geom1/geom2`, position, frame, and signed distance); `mj_contactForce` can provide per-contact force. The RoboSuite environment also exposes contact checks, and its grasp helper returns a Boolean. Those are useful raw signals, but the drill-place success predicate does not call a grasp helper or count finger contacts.

The live object state is available from the free-joint qpos/qvel and derived world body pose (`body_xpos`, `body_xquat`) and velocities (`get_body_xvelp`, `get_body_xvelr`). The playback loop also reads the whole flattened simulator state for fidelity comparison. Those values are sufficient to log the object trajectory, but playback does not currently turn them into stage-by-stage acceptance evidence.

## Current success predicate

For `LMDrillPnP90`, `_check_success()` delegates to `scene.success()`, which evaluates an `AllCriteria` conjunction:

| Criterion | Exact check |
| --- | --- |
| `IsStatic(drill)` | Object linear speed `<= 0.002 m/s` and angular speed `<= 0.005 rad/s`. |
| `IsUpright(drill, 0.95)` | The object's local +Z axis has world-Z alignment `> 0.95`. |
| `IsGripperFar(drill, 0.1)` | Both left and right end-effector sites are more than `0.1 m` from the drill root-body position. This is a point-to-point distance check, not a no-contact check. |
| `IsInContact(drill, target_table)` | RoboSuite reports any contact between the drill object and target table. It has no target-region or placement-pose test. |

There is no grasp, lift, source-table departure, transfer-path, release, target-region containment, settle-duration, finite-state, penetration-depth, or state-write-provenance criterion. The predicate is evaluated as an instantaneous Boolean, without a dwell interval. It does not inspect active equality constraints.

**Therefore, yes:** a task can report success after an object teleport if the resulting state meets these four checks; it can report success with excessive penetration if MuJoCo still reports target-table contact; and it could report success while a constraint holds the object still. The audited task itself does not create an object weld, but the success predicate would not detect one introduced by another model/runtime layer. It can also pass after the object was pushed or slid onto the target without a grasp-and-lift sequence.

## Proposed machine-checkable M0 gates

Use an explicit configured task geometry and thresholds, then run these as an ordered per-episode state machine. Values below are starting defaults for configuration, not constants tied to the drill. All durations are measured in simulation time and converted to substep counts using the actual physics timestep. The terminal M0 result is PASS only if every required stage and the global invariants pass.

| Gate | Machine-checkable rule | Example configurable defaults |
| --- | --- | --- |
| **GRASP** | Simultaneous positive-force object contacts from at least two configured, distinct digit groups, including the thumb, continuously for `grasp_hold_s`. Record the contacting geom IDs and forces. | `min_digit_groups=2`, `thumb_required=true`, `min_normal_force_n=0.1`, `grasp_hold_s=0.10`. |
| **LIFT** | After GRASP, object lowest collision point rises at least `lift_height_m` above the configured source support plane for `lift_hold_s`; source support contact is absent while the grasp remains. | `lift_height_m=0.05 m`, `lift_hold_s=0.10`. |
| **TRANSFER** | After LIFT, projected object-center progress along the configured source-to-target vector reaches `transfer_progress_min`; source support stays absent. Progress is computed from recorded poses, not inferred from the final state. | `transfer_progress_min=0.80`; task geometry supplies the vector/target frame. |
| **RELEASE** | After TRANSFER, all configured hand-object contacts are absent for `release_hold_s`, the object is supported by the target, and the hand clears the configured release distance. | `release_hold_s=0.10`, `release_clearance_m=0.05 m`. |
| **FINAL PLACE** | Object footprint is inside the configured target polygon to `min_footprint_fraction`; bottom-to-support-plane gap is within tolerance and orientation is within the configured target tolerance. | `min_footprint_fraction=0.90`, `support_gap_tolerance_m=0.01`, task-configured angular tolerance. |
| **SETTLE** | FINAL PLACE remains true for `settle_window_s`; target support contact remains, hand contact stays absent, object linear/angular speeds and total pose drift stay below configured limits. | `settle_window_s=0.50`, `linear_speed_mps=0.01`, `angular_speed_rps=0.05`, `pose_drift_m=0.01`. |
| **FINITE STATE** | Every physics substep, all recorded simulation time, qpos, qvel, controls, actuator outputs, object pose/velocity, contact distance, and measured contact-force values are finite. Fail on the first non-finite sample. | No numeric tolerance; optional list of logged channels is configuration. |
| **MAX PENETRATION** | For enabled physical collision pairs, `max(max(0, -contact.dist)) <= penetration_limit_m` each substep. Record the worst pair and time. Exclude only explicitly documented nonphysical/sensor geoms. | `penetration_limit_m=0.005 m`. |
| **NO TELEPORT** | Whitelist the initial reset and the single `states[0]` restore. After that checkpoint, object free-joint qpos must be unchanged between controller output and entry to each physics substep; only MuJoCo integration may change it. Check every substep and enforce configurable linear/angular velocity bounds on pose deltas. Any reset/state-restore/object-qpos write outside the whitelist fails. | `max_linear_speed_mps` and `max_angular_speed_rps` are task/object configuration; use a small configured numerical epsilon for pose-delta consistency. |

Keep the gate state, first-passage simulation time, substep index, threshold values, observed extrema, and failure reason in a machine-readable JSON record. Save per-substep CSV/Parquet only if the complete run log is too large for JSON. `NO TELEPORT` must be enforced by instrumentation around the control-to-physics boundary plus write-path auditing; a final-state comparison or a large-jump heuristic alone is not proof that no qpos write occurred.

Suggested state progression:

```text
RESET_CHECKPOINT -> GRASP -> LIFT -> TRANSFER -> RELEASE -> FINAL_PLACE -> SETTLE -> M0_PASS
      |                |       |         |          |             |          |
      +----------------+-------+---------+----------+-------------+----------+--> FAIL
```

FINITE STATE, MAX PENETRATION, and NO TELEPORT are global invariants checked from the reset checkpoint through SETTLE; any failure transitions to FAIL. A stage timeout also fails the episode. Thresholds, target/source geometry, hand digit grouping, and timeouts belong in a task configuration file so another object/task can reuse the evaluator without copying drill-specific rules.

## False-positive risks and interpretation limits

- `final_success=true` currently proves only that the final sampled state meets four simple checks. It does not prove that any earlier grasp/lift/transfer/release events occurred.
- Any drill-target-table contact counts, even if it is at the table edge or a collision mesh is badly penetrated. There is no placement zone check.
- `IsGripperFar` measures only the distance from each end-effector site to the drill root-body origin. The drill can still touch a finger or another robot part while the root-origin distance exceeds `0.1 m`.
- A pushed/slid object, an object restored to a convenient state before checking, or a constraint-held object may satisfy the same endpoint predicate.
- `final_success` and demonstration-state fidelity are independent outputs. With divergence allowed, a task-success bit may be true despite a state mismatch. Even a matching recorded state sequence does not itself prove the intended grasp contacts, stage ordering, or absence of an injected state unless the active stepping/write provenance is recorded.
- This is MuJoCo simulation evidence only. It does not establish real G1 hand contact performance or Sim2Real transfer.

## Evidence for a near-trivial final human visual review

Provide one reproducible artifact bundle per episode with:

1. A short, unbroken, timestamped video from synchronized side and overhead views, with the source and target support planes/regions overlaid. Mark GRASP/LIFT/TRANSFER/RELEASE/SETTLE transitions and show the object and finger collision geoms or contact points.
2. A compact live HUD showing object pose, linear/angular speed, hand-object contact count and normal-force summary, target-region fraction, maximum penetration, and the NO TELEPORT monitor status.
3. The JSON stage report and per-substep trace, exact task/configuration thresholds, physics timestep, random seed, model/dataset/controller hashes, and run command. Include source-state restore time and every reset/write-provenance event so the one allowed initial state restore is visually and mechanically distinct from active playback.
4. A final still with object footprint, target polygon, support contact, and settled-speed values, plus a plot of object height and planar path over simulation time.

This lets a reviewer look once for visible geometry mismatch, implausible penetration, or a mismatch between the contact/phase trace and the video while relying on deterministic gates for the repetitive measurements.

## Source references

All upstream links below are pinned to `d82844dcec242c82d6b82628ccc45933c3ad5cbd`:

- [DrillPnP task and endpoint criteria](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/envs/locomanip_pnp.py#L270-L325)
- [Scene success delegation and initial placement writes](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/utils/scene/scene.py#L137-L161)
- [Dynamic object free-joint and collision parameters](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/models/objects/objects.py#L53-L109)
- [Power drill collision and visual geoms](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/models/assets/objects/omniverse/locomanip/powerdrill_b01/model.xml)
- [Success criteria implementations](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/utils/scene/success_criteria.py#L82-L175)
- [WBC-goal playback state restore, active stepping, and final bit](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/dataset_playback.py#L329-L415)
- [Per-episode playback loop and result aggregation](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/dataset_playback.py#L491-L574)
- [Initial state restoration and success wrapper](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/wbc/envs/locomanipulation/sync_env.py#L138-L188)
- [G1 three-finger hand configuration and contact geom groups](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/models/grippers/g1_threefinger_hands.py)
- [WBC joint-action handoff to the simulator](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/wbc/envs/locomanipulation/utils/locomanip_env.py#L311-L328)
- [Playback mode distinctions and source-demo command](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/README.md#L110-L130)
