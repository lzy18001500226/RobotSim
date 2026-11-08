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

## Current delivery Goal (2026-11-08)

Reuse the existing runner and task interfaces:
START -> known table -> RGB-D bottle observation -> reachable grasp or bounded
necessary stance correction -> actual lift -> collision-screened carry posture
-> held-object locomotion to the known chair area -> measured stable stop.
Chair/table coordinates may use pre-mapped scene truth. Chair-seat placement
requires supported reachability and real seat support; holding at the chair
area is a separate result. Do not invent a squat, support or attachment.
Do not require all four precision micro directions before a qualified route.
After the truth-base chain passes, qualify LiDAR pose substitution and port the
verified runtime to existing C++/ROS owners. Reserve a voice task-goal interface.
No new framework/worktree, robot mechanics or vendor binary/policy edits.

Native-337 zero-step geometry found chair body raicom_table_chair at (0,1.5,0),
seat z about 0.328215m, but whole-mesh convex collision top at center 0.536048m.
Do not mistake the 208mm hull gap for successful seat support or a robot
crouching limitation. Source scene and robot remain unchanged.
Evidence: reviews/issue-46/fetch-delivery-chair-geometry-20261107T173112Z/.

### Explicit bounded carry integration

The carry adapter reuses PointGoalController, the existing native ROS transport,
and the unchanged composite stop evaluator. Its explicit diagnostic profile is
SIMULATION_ONLY_FETCH_CARRY_ROUTE: task motion deadline <=90s, input forward
<=0.12m/s, angular <=0.075rad/s, native root <=0.20m/s, original 29-axis source
bounds, telemetry age <=0.20s, goal-base tolerance 0.15m, >=1.05s zero publishing
before STAND and full original composite stop. Only actual intended distal-pad
bottle contacts are allowed. A separate grip owner continues HAL arm/hand
control. Carry reuses the circle IK with a pelvis-heading-relative TCP and
world-upright axis at 50Hz, including source-bounded joint-velocity feedforward.
A carrier fault revokes navigation health before a <=5s last-command grip hold.
No force or object attachment is introduced.

The current chair-area test goal is base-world (-0.45,1.0), derived from the
known chair at (0,1.5), not the chair CAD center itself. This point-goal task is
not an old micro pulse or straight/curved-reference tracking qualification.
Raw cross-track diagnostics remain visible; room workspace and actual contact
guards apply. Default micro and reference-path thresholds are unchanged.

Separate explicit diagnostics retain the default profile above:
--carry-sdk-minimum-turn uses the pinned SDK example's 0.10rad/s angular minimum;
--carry-native-action-smoothing adds native ActionSelector SD/LD 500ms rules;
--carry-native-step-mode selects the installed unchanged CPGStep policy/config
and adds matching SD/LS rules. This is not FoundationWalk qualification.
A successful LS service selects STEP speed_mode=4 while zero input can retain
SD/100; that is preparation only. Actual source and LS readback are still
required after effective input. Source and target stable-stop criteria stay
unchanged. --carry-grip-force-margin=3 raises only the carry force objective,
after proven lift/retract, not the existing 2Nm actuator cap. With the approved
local-friction comparison, the same explicit margin is preloaded before lift;
real sustained pad force must meet that budget before the probe begins.

Current physical result (2026-11-08): RGB-D lift/retract and actual8-second
static carry passed with the specifically approved local friction reference and
NoSlip0; chair delivery has not passed. Native LS/source and its 65-element
policy observation were observed, followed by measured root-speed violation.
NoSlip=0 with approximately3N normal per pad instead showed about0.436mm/s axial
soft-contact creep and failed the10mm hold guard before motion. No target was
reached. Do not rerun an identical failed profile.

A specific pad/bottle-only friction-reference diagnostic is specifically approved (2026-11-08) and implemented for validation: 22 already enabled geom pairs, same geometry/mu/normal parameters,
solreffriction inherited0.02s ->0.002s (=2*native timestep); globalNoSlip stays0.
The exact direct approval is recorded in model_integrity.md C.2; no permission
is generalized to other contact values or limits. See the measured findings for its
exact source hashes, proposed pair table and approval status.

Source chair connectivity contains14 separate solids; the seat component's top
at(0,1.5) is0.328213m. No scene collision changes have been implemented.
Three finite fixed-stance and bounded/full-source waist diagnostic solves did
not establish a placement solution. They do not prove global unreachability or
qualify a crouching controller. Keep placement NOT_RUN.


Only a proven host-clock epoch fault can enter bounded zero/confirmed-STAND/
complete-stop recovery before a new command; maximum four, no task-deadline
reset. Physical faults, protection, stale state and publisher conflicts cannot
rearm. A full recorded native trace audit independently checks limits so
consumer-skipped samples cannot silently pass.

## Historical five-stage Goal and observed status (2026-11-07)

The maintainer authorized sequential autonomous completion of this simulation
Goal on 2026-11-07, with Stage 1 taking priority. Reuse this worktree, task state
machine and evidence roots. Routine implementation, official-source research,
offline tests and evidence-driven bounded simulations need no per-run approval.
This does not authorize mechanical redesign, hardware or success overrides.

1. **Truth-first physical grasp:** fixed table stance, actual open/approach,
   measured circle/axis arrival, bilateral force-bearing closure, >=50mm free
   bottle lift and >=1s unsupported hold. Independently reproduce before claiming
   a working full start-to-table-to-grasp review command.
2. **Reachable stance region:** evaluate nearby x/y/yaw and supported arm/waist
   postures, collision margins and balance. Include 300mm and 150mm bottle-height
   cases as separately identified scene inputs; do not promise every pose works.
3. **RGB-D substitution:** use fresh calibrated bottle axis/diameter/section pose
   in the same execution chain. Record estimator error against truth separately;
   truth must not silently feed perception-qualified control.
4. **Perception-driven stance adjustment:** at a known table with unknown bottle
   location, observe, assess reachability, make only qualified bounded base
   corrections, stop, reobserve and replan. Stale pre-move coordinates are invalid.
5. **LiDAR substitution:** replace base-truth localization with the established
   mapped LiDAR estimator, retain local RGB-D grasp feedback, and verify the
   complete reproducible chain. Delivery/manipulation after lifting is out of scope.

Keep each stage's physical result separate. A plan does not qualify a stage.
Preserve existing geometry and force/posture/stop constraints; use
the scoped parameter authorization below for ordinary gripper diagnostics.

The five numbered stages above are the current Goal contract. Older task
messages used “Stage 2” for fixed-stance manipulation and “Stage 3” for base
repositioning; those shorthand labels do not override this five-stage audit.
Fixed-stance grasp and RGB-D subtests are qualified separately from the
strengthened walking safety audit below. Repositioning reuses that grasp chain,
requires qualified bounded micro motion and a fresh observation after stopping,
and cannot reuse pre-move object coordinates.

The physical grasp/lift observations below remain valid, but the new native
source-speed audit changes the **full-chain qualification**: previous runs did
not enforce native leg speed continuously while walking. Recorded walking
peaks exceed the installed 11.936 rad/s knee bound. Updated runs fail closed
on native body/leg speed; the strengthened START-to-table chain is not yet
qualified. Do not present historical grasp PASS as passing this new audit.

Stage 1 has one historical measured full START-to-table-to-lift PASS (native run
`20261106T210324Z`): final dock error 10.6mm, minimum bottle rise 55.60mm over
51 samples spanning 1.000s, zero table support, and bilateral lower-concave-pad
contact. An independent fixed-stance run also passed (58.49mm minimum rise).
This uses the explicitly selected **SIMULATION_ONLY source-derived CoACD pad
comparison**, zero internal joint friction, right-claw kv=1.27, and elliptic
cone/impratio=10. It does not qualify the untouched whole-mesh collision
baseline, manufacturer-default physics, perception, or the subsequent stages.
Independent full-flow repetition `20261106T212727Z` also passed (57.10mm minimum
rise, 51 unique samples over 1.000s, no table support, cleanup confirmed).
See the [measured findings](../research/issue46_truth_grasp_execution_findings.md)
for exact trace identities; do not promote a prior model's PASS to this setup.

Canonical RGB-D substitution now has a fixed-stance physical PASS
(`20261106T234215Z`, 57.82mm minimum rise) and a full START-to-table physical
PASS (`20261107T000115Z`, 57.52mm minimum rise). The full run uses a fresh
calibrated virtual head RGB-D observation, independent pre-execution
reobservation (0.093mm target difference), and an explicitly bounded stationary
object snapshot. It is not continuous visibility through the grasping hand.
Both achieved 51 samples/1.000s unsupported hold; all audited forces fall on
the approved lower arcs. Assumptions remain upright blue cylinder, known
height prior and the explicit source-derived collision/control profile above.
Base truth and independent truth safety/audit are still used; LiDAR substitution
and perception-driven base adjustment are not qualified by these results.

The 150mm height variant now has source-waist-roll **geometric** candidates
(050007Z): approach/lift/contact and static support screens pass, with >=39.66mm
COM support margin. This uses an existing DOF, no mechanical change. Actual
waist handoff, dynamic balance and short-bottle grasp remain unqualified.
The 0.12m/s, 1s left micro-step (035847Z) passed measured stopping but has coupled
forward motion; general forward/back/left/right correction is not yet qualified.

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
successfully today. The research lift executor has the narrowly qualified
simulation result above; delivery remains unimplemented/unqualified. Every
transition is fail-closed. A failed localization, stale frame transform,
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
   default remains waist roll within 2 degrees for the qualified 300mm baseline.
   The current maintainer Goal explicitly includes supported bending/side-body
   posture for shorter bottles: separately labeled posture candidates may use
   the existing source waist-roll DOF within its original limits. Require
   collision, support/balance, smooth approach and actual controller handoff
   qualification; do not mistake the ordinary preference for a hardware limit.
   Do not add joints, alter axes/limits or execute an unqualified posture.
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
   Actual full opening has passed in the labeled zero-internal-friction overlay;
   this is not vendor-default parameter or complete grasp acceptance.
6. Only after the physical pregrasp remains stable and the actual hand is in
   the screened contact region may the gripper close.  Require sustained
   bilateral distal-pad contact before attempting lift.
7. Lift in order: a small clearance probe, then a 50 mm lift and hold.  A
   grasp PASS requires table-support transfer, opposing pad contact, object
   COM rise of at least 50 mm, and stable hold.  No weld, object qpos write,
   hidden support, or collision exclusion may substitute for those checks.

Failure handling remains local to the failed stage:

- Static-screen failure: adjust only base stance, supported object placement,
  source-supported knee/waist posture, or arm/wrist approach pose.
- Standing or pregrasp failure: correct the initial state or the smooth
  transition before issuing a close command.
- Hand-to-object error or missing bilateral contact: search the screened
  stance/IK neighbourhood using the measured MuJoCo hand pose; do not merely
  increase gripper force.
- Contact or lift failure: inspect contact forces, table support, root motion,
  and object COM before trying the next bounded candidate.

Each failed candidate must preserve its result and select the next local
candidate. Under the current sequential autonomous Goal, a recorded grasp
success proceeds to the already authorized next stage; it is not a routine
permission stop. Stop for an actual required visual decision, an unapproved
source/model change, or a failure outside supported controls after completing
independent useful work. Apply the
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


### Empty-bottle transport criterion (maintainer clarification, 2026-11-08)

The current empty H300 bottle may tilt during transport; world-upright attitude
is an observation, not grasp failure after a qualified lift. The explicit
--allow-carried-bottle-tilt profile preserves the legacy strict profile and
records bottle angle. It does not alter the desired IK task, source mechanics,
materials, actuator limits or robot posture limits.

Keep the existing grasp-point displacement limit10mm, bilateral contact-loss
window0.10s, drop-height/environment-support checks, source/MC freshness and
the complete physical/controller-target stop gates. Initial upright perception
and pre-closure motion checks remain in force. A published close command is not
proof of retention; old strict-upright failures are not relabeled as passes.
