# Issue #46 manipulation integration prototype

This directory is a **research-only, transport-neutral** prototype. It contains pose/time/frame contracts, a docking handoff validator, Cartesian tool-approach utilities, a source-limit-aware IK witness validator, and a fail-closed static gripper geometry gate. Runtime, ROS, Local03 navigation, Local04 mechanism, and Candidate B source files are not dependencies.

## Validation boundary

- `make_linear_approach()` builds an explicit straight Cartesian path from a caller-selected approach direction. It does not decide whether top-down insertion or a side approach is better.
- `validate_ik_waypoints()` checks ordered joint names, source limits/margins, IK residuals, progress, and optional per-sample joint steps. It validates an IK witness; it does not prove dynamic arm tracking.
- `evaluate_gripper_feasibility()` requires a fresh object pose, a complete model-identified collision-pair inventory at each path sample, and all enabled gripper/object pair checks during closure. It rejects missing/incomplete geometry, collision/clearance failures, unavailable bilateral pad reach, out-of-range joints, wrong object-local contact band, unverified open/close range, and linkage-first interference. Contact-band checks project recorded pad contact points onto the supplied object-local axis; they are not caller-supplied booleans.
- `PASS_STATIC_GEOMETRY` means only a complete static geometry witness passed the supplied thresholds. It says nothing about contact force, friction, stable hold, table unloading, lifting, transfer, release, placement, or settle.
- Passive/follower values in `JointCommandProfile` describe evaluated kinematic state only. They do not authorize direct follower actuation or qpos writes. `GripperContract` separates commanded joints from derived joints and records the constraint name for each derived joint; derived coordinates need not have independent actuator limits. The contract requires a range-evidence reference and explicit static-range verification status.

## Pose, time, and frame contract

`PoseSample.frame_id` is the frame in which its position/orientation are expressed. Quaternions are normalized `xyzw`; spatial units are SI. The acquisition stamp names a `TimeDomain` (`SIMULATION`, `ROS`, or `SENSOR`); it is never compared with `received_steady_ns`, which is local monotonic freshness metadata.

- Simulator truth uses `PoseSource.SIM_GROUND_TRUTH` and exact model identity; it remains distinct from a measured or estimated pose.
- RGB-D object results should be measurements in a named optical frame such as `camera_color_optical_frame`, stamped at image acquisition, with a calibration ID, generation/sequence when available, and local steady receipt time. A detector is not implemented here.
- Convert optical-camera measurements through explicitly ordered `T_parent_child` edges at the acquisition time. Dynamic edges must use the same clock domain and exact stamp. Static calibration edges are timeless. This API intentionally has no inverse lookup or “latest transform” fallback.
- Adopted world chain semantics remain `map -> odom -> base_link`; localization owns `map -> odom`, odometry owns `odom -> base_link`. Camera optical axes remain REP-103 optical convention.

## Navigation docking handoff

`evaluate_docking_handoff()` consumes a navigation completion state plus measured/ground-truth base pose, explicit target pose in the same frame, measured base twist, generation/sequence, and observed stopped dwell. It rejects non-success/cancelled navigation, stale pose, source/epoch/frame mismatch, position/yaw tolerance failure, motion, or insufficient dwell. Commanded-velocity integration is not accepted as pose. The existing Issue 80 adapter at `/home/lzy18001500226/robotsim-issue80-offline-pointgoal-20261020/ros2_ws/src/robotsim_navigation/` consumes Odometry and PoseStamped, but those message contracts do not themselves carry the epoch/sequence fields required here. A reviewed lifecycle metadata bridge is still needed; this prototype does not synthesize one or touch Local03.

## Candidate B freeze and existing result

The tested Candidate B source geometry remains frozen at SHA-256 `b484595375381e4945d3685bf2b2346786b634c6a13b7433f7dfcfbd21252d22` and classification `UNVERIFIED_EXPERIMENTAL`. The screen also records its base runtime MJCF as `7a54a3c982779ca934a64fff9fdfe99bbb91460cac8d394be3df0aa7b52fc205`, integrated geometry-selected MJCF as `65a5cd5d0faa12c8b31c0ae478952c5f5120f7ae56f105e450fc9b8ae165f41f`, and full-X2 fixed-base static diagnostic model as `0eadbaf8dff1968c5cab2d9892c140107368d0bc66b835d67df1da79037bad01`. Its recorded collision-exclusion configuration SHA is `ef513d5d9fb4dd85f726d1c1868b4c6be2d359eee1ccea20c0c4eb2c816fddb6`. MuJoCo 3.3.6 compiled counts are 45 bodies, 40 joints, 252 geoms, 5 equalities, and 32 actuators. The frozen static model has 137 enabled right-gripper collision proxies. CAD values `[0.20, 0.55]`, open `.55`, and close `.30` are evidence-specific simulation inputs, not a vendor command contract. The audit output freezes source input hashes, all 137 proxy names, and prior configuration outcomes.

The prior finite screen already tested 25, 30, 60.3, 65, and 70 mm objects and two height offsets. No candidate passed. This implementation does not repeat that screen. The new evidence audit reuses exactly one existing 70 mm full-X2 static witness: IK targets pass the saved 2 mm / 1 degree / 0.05 rad gates, but the first pregrasp-hover sample already has loop/object contacts (signed minimum about -1.051 mm), and the sampled path reaches about -8.315 mm loop/object distance and 0 mm hand/workcell clearance. At CAD `.30`, both pad proxies touch while linkage and other-hand proxies also penetrate; the saved branch reports loop contact at `.55`, before bilateral pad proximity at `.305`. Thus the route is statically blocked; pad closure is not a grasp.

The earlier dynamic log is read-only context, not reproduced here: at 9.165 s the right shoulder pitch target was -0.932487 rad and actual -0.884434 rad (0.048053 rad lag); the recorded hand vertical support was negative. No new arm movement, grasp, or lift is claimed.

## Reproduction

Run focused interface tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/research/issue46_manipulation_integration/tests -t .
```

Audit the saved, single-configuration full-X2 evidence under the pinned MuJoCo 3.3.6 environment:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_saved_candidate_b.py
```

The audit verifies the source evidence manifest and hashes, compiles the saved MJCF to check model identity and joint limits, validates the stored IK target sequence, and reconciles its collision CSV. It invokes no `mj_forward`, `mj_step`, `mj_step1`, or `mj_step2`; it changes no model file and no qpos. A new unique evidence directory is created under canonical `reviews/issue-46/` with `REPORT.md`, `result.json`, `REPRODUCE.md`, and `artifact_paths.json`.

## PR #90 gripper in the full-X2 static preflight

The maintainer-reported isolated unloaded dynamic validation for PR #90 used MuJoCo 3.3.6 for 17,500 steps, completed 2% HOLD -> 100% OPEN -> HOLD -> 2% CLOSE -> HOLD, and measured a 90.999 mm open five-section minimum, 12.831 mm closed-side minimum, and 0.015576 rad maximum root tracking error against the unchanged 0.02 rad limit. This establishes isolated unloaded opening/closing only.

The full-X2 bottle feasibility audit is separate and remains BLOCKED_STATIC:

- Exact PR #90 head facdaa0d6e5f8152168d2e812d864b660eac9f96 and isolated MJCF SHA 32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45 were used read-only.
- The old full-X2 model SHA is 0eadbaf8dff1968c5cab2d9892c140107368d0bc66b835d67df1da79037bad01. The generated static overlay SHA is fad9ae69ba99f1c8e5efd00f78a9571e9ccc04f48c7a10513ae711187718c302. Only the two q2 equalities changed.
- PR #90 and old/new full-X2 collision proxy inventories have identical 137 collision-mesh hash records. The official right wrist mount and 32-actuator full-body model are preserved.
- The 70 mm by 230 mm, 0.550 kg upright cylinder's top-down path is blocked at the first hover sample by R_hand_wide_loop_Link__wide_loop_Link_coacd_02 against test_cylinder_collision at -1.051 mm. The sampled path minimum is -8.315 mm.
- An existing side/oblique full-X2 route is also blocked by narrow1-to-wide3 self-clearance of 0 mm.
- With the gripper fully open at the selected grasp pose, linkage-loop clearance is -8.299 mm and other-gripper clearance is -2.729 mm. No bilateral pad proximity lies within the 10 mm mid-body band; the loop is already intersecting before closure. A closed-end narrow3/wide3 collision reaches 1.752 mm penetration.
- A bounded five-variant static IK screen also failed to find a clear path: +20 mm contact height and +15 degree wrist yaw had source-valid IK but collided; -20 mm contact height, -15 degree yaw, and a 50 mm higher hover did not produce an IK witness under the source limits and bounded solver. This is not a proof that no other pose exists.
- Physics steps remain zero for both full-X2 static screens. GRASP, LIFT, TRANSFER, RELEASE, PLACE, and SETTLE remain NOT RUN.

Full-X2 evidence:

- Main audit: /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-quartic-full-x2-static-20261023T144705Z/
- Bounded pose screen: /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-bounded-pose-static-20261023T143942Z/

Reproduction commands:

    cd /home/lzy18001500226/robotsim-issue46-manipulation-integration-local02-20261023
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_pr90_full_x2_static.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_pr90_bounded_pose_variants.py

These are static IK and geometry checks using mj_forward and collision-distance queries. No bottle-grasp dynamics were launched. PR #90 and PR #91 remain separate, open Draft review branches; Issue #46 remains open.

## Resume conditions

Before any new physical grasp trial:

1. Local04 provides a reviewed corrected gripper model and its actual open/close joint profile, limits, collision inventory, and model SHA.
2. Recompute a full-X2 PREGRASP/APPROACH from a ground-truth or RGB-D-compatible object pose and run the complete enabled-pair geometry gate. An alternate approach is allowed only if its sampled path is collision checked against the current geometry.
3. Resolve the first obstruction and shoulder trajectory-tracking risk without relaxing source safety limits.
4. Only after static gates pass, request/perform a separate authorized dynamic arm and physical-grasp validation. Keep `GRASP`, `LIFT`, `TRANSFER`, `RELEASE`, `PLACE`, and `SETTLE` statuses separate.

Prototype disposition for this task: `KEEP-AS-TOOL` pending focused review; this package is not imported by production or ROS runtime code.
