# Truth grasp execution: opening and pose findings

Current checkpoint, 2026-11-11: both H300 and H220 have one completed
START-to-second-table physical transfer witness in the native MC/MuJoCo runtime.
This is a SIMULATION_ONLY D70 cylinder baseline at 0.200 kg. It does not establish
the historical 0.570 kg bottle requirement, hardware fidelity, or broad repeatability.
Human visual review remains pending. Historical failures below are preserved.

## H220 complete START witness

`reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261111T004845Z/`
completed real START approach, fresh simulated RGB-D grasp, lift, retraction,
native standing/waist recovery, retreat/turn/carry, table contact, release,
80 mm open-hand withdrawal and independent table support. Every corresponding
runtime gate passed; no terminal failure; native container absence was verified.
The initialized root was (-1.5, -1.5, 0.65192435) m, not a table-front replay.

| Measured quantity | Result |
| --- | --- |
| Cylinder | D70 mm / H220 mm / 0.200 kg |
| Initial unsupported lift | min 59.003 mm / max 60.837 mm; continuous qualified 1 s |
| Minimum bilateral pad normals | 13.9991 N / 14.0017 N |
| Mean upward hand support during lift | 1.962279 N; pickup-table support 0 N |
| Whole-episode COM peak rise | 208.951 mm, including native standing recovery |
| Carry base endpoint displacement | 2.56845 m |
| Actual dock error | 107.48 mm; 0.267891 rad heading error |
| Released independent support | 2.02 simulation seconds; minimum table support 1.961884 N; pad force 0 N |
| Final bottle orientation | upright error 0.03736 degrees |
| Native records | 15,203, continuing beyond the old 12,000-record cutoff |

A separate fixed-table H220 witness `raicom-stl-xvfb-native-reset-stand-20261110T235520Z`
also passed release/withdraw/free support. These are two different starts, not a
claim of statistical repeatability. The tested START scene source SHA-256 is
`297ac89ef0d807cee105a29c19991fa0c70838475c8425dba22e0893feb27f87`;
the approved gripper geometry, joints, coupling, collision masks, friction, gains
and actuator limits were preserved. Runtime physics was native MuJoCo 3.3.7;
host 3.3.6 was used for offline IK and review rendering.

The task no longer makes a separate final base-heading correction. Functional
arrival remains separate from the retained 150 mm / 120 mrad diagnostics; actual
stopped-pose arm IK and collision screening must pass before placement motion.
The user-requested retraction remains, without the extra preset inward wrist
stow. Placement uses a bounded source-range tabletop/arm IK and a source-valid
small tilted contact posture; actual table support gates release. The freely
released cylinder settles upright under contact/gravity physics. A planned tilt
or an IK result is never counted as placement acceptance.

Angle units were corrected: `axis_grasp_error` stores `0.1*(axis-target)`.
Its norm is a scaled chord; the actual angle is `2*asin(norm/.2)`. The placement
solver and tracker now use the same 5 mm position / 1 degree axis-tracking target
contract around the selected posture. No joint range or physical safety limit
was enlarged. Older informal “1 mrad/1 degree” interpretations of raw residuals
must not be used to upgrade any earlier failed static or dynamic study.

### Confirmed observer lifetime defect

The old read-only interposer stopped emitting after 12,000 records while still
calling the real `mj_step`. The three late failures 231041Z, 232044Z and 233127Z
all contain exactly 12,000 rows; the prior H300 successful run had 11,500 rows.
This created stale-state errors without proving a physics freeze or vendor defect.
The lifetime cutoff was removed; controller motion budgets, sampling cadence,
200 ms state acceptance and every physical gate remain unchanged. Temporary
late-snapshot retry code was removed after this root cause was identified.

`native-observer-12000-frame-regression-20261110T234818Z` is an executable C
interposition regression using exact 3.3.7 headers and a stub step, not dynamics:
12,020 calls produce 12,000 old records and 12,020 fixed records. The actual
235520Z and 004845Z runs then independently exceeded the old cutoff and completed
physical placement. The fixed observer source SHA-256 is
`c450cff4c4c06f84e048fa7b5788c538ecb4cc20bfba80272965cdf4b9766f01`.

### Evidence and reproduction

Local canonical evidence is under `/home/lzy18001500226/projects/RobotSim/reviews/issue-46/`:

- measured summary: `h220-full-start-transfer-closeout-20261111T010140Z/measured_result.json`, SHA-256 `acbd3c6cb14971c1f09c95f530c5a48eee9f168b28e9d96b347bc7b8d9beeb85`;
- source episode result: `004845Z/result.json` within the full run directory above, SHA-256 `8c86d613bf3d1918633261ad222130214cd7cade66f8a90b98fb3e5307951199`;
- full native trace SHA-256 `e18e8081bbeb8a557e4316419ddac292cd2833fb5d423dbc7ac840a4dc96d8eb`;
- complete review MP4: `native-transfer-recorded-review-20261111T005652Z/full_x2_actual_transfer_review.mp4`, SHA-256 `43fc26622df7880672599401653eaca73f519ad9d8efba8af41b9830efb43fd3`.

The MP4 renders measured real native-physics states in the matched full model;
it sends no commands and performs no new stepping. It is not the force/lift proof;
raw native contacts, motion and released-support gates are the proof. Local large
artifacts are preserved, not committed as Git blobs. Linux and Windows UNC access
were checked by matching SHA-256, and the entire MP4 decoded without errors.

Exact locally executed PowerShell command (requires the existing pinned runtime,
local source inputs and offline environment; not a clean-clone installation claim):

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec env DISPLAY=:0 PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python -B scripts/research/issue46/grasp/live_retreat_turn_review.py --to-delivery-table --from-start --bottle-height-m .22 --multiccd
```

Stop at human review. Keep PR #106 Draft and Issue #46 open. No new dynamic test,
main merge, model promotion, vendor edit or local cleanup is authorized by this result.
The research implementation remains isolated and has disposition **PORT-TO-C++**.

## H220 fixed-bottle stance review (2026-11-11, static only)

The bounded source-preserving study sampled175 support-anchor/yaw combinations
at the original pickup table, with D70/H220mm/0.200kg bottle pose fixed.53 have
saved sampled static preparation/reach/approach/closure/60mm-lift witnesses;
43 have endpoint IK but a checked path/contact obstruction, and79 remain
solver-unresolved. These are not global infeasibility or physical-grasp results.
All positive witnesses use the existing pre-raised/crouched posture family.

At yaw offsets -30/-15/0/+15/+30 degrees the positive counts are1/12/14/15/11
of35 samples. The zero-offset positive envelope has holes; it is not a certified
rectangle or a continuous tolerance region. A small gripper-linkage/bottle gap
must not be misrepresented as a required base-position accuracy.

The final static states evaluate the unchanged frozen MJCF joint polynomials,
not the helper CSV interpolation. Independent readback checked4280 positive
path frames: no limited-joint violation, maximum equality residual2.776e-16rad,
source XML/model arrays unchanged. No mj_step, MC, walking or dynamic grasp ran.
COM support stability, visibility and continuous swept clearance are not certified.

Review evidence: `reviews/issue-46/h220-stance-region-static-exact-coupling-20261111T084002Z/`;
interactive `index.html` includes yaw/endpoint-versus-path selection and exact
pose renders. Summary and artifact hashes are in
`h220-stance-region-review-closeout-20261111T085745Z/` under the same evidence root.
Reproduce with `stance_region_static_review.py`, `render_stance_region_review.py`,
and `verify_stance_region_static.py` under `scripts/research/issue46/grasp/`.
Disposition: RETAIN_RESEARCH for human review; not a production dependency or a
promotion of the frozen SIMULATION_ONLY hand. PR106 remains Draft; no main merge.

## Earlier checkpoints (historical)

| Actual native episode under reviews/issue-46 | Observed result |
| --- | --- |
| raicom-stl-xvfb-native-reset-stand-20261109T204903Z | Fixed-table RGB-D grasp, lift, retreat/turn, STEP carry, release/withdraw and independent table support passed. |
| raicom-stl-xvfb-native-reset-stand-20261109T221003Z | Complete aligned START pickup navigation and the same physical transfer passed; minimum final table support force 1.9537 N, zero pad force for 2.0 simulated seconds. |
| raicom-stl-xvfb-native-reset-stand-20261109T231915Z | Actual carry progressed 2.5368 m and composite stop passed, but the legacy 0.15 m approach gate failed at 0.1996 m; placement was NOT RUN. |
| raicom-stl-xvfb-native-reset-stand-20261110T015906Z | H220 native lowering, torso yaw/forward bend and composite stand passed. RGB-D image-aspect heuristic rejected a visible bottle; no arm approach or closure. |

The functional transfer profile now screens placement from the actual stopped
pose, uses a 0.25 m approach allowance separately from the retained 0.15 m
precision result, a 0.16 m retract and a delivery section nearer the table edge.
Those changes do not establish a new full-flow repeatability result.

For H220, the intended grasp section is 0.12 m above the bottom (0.01 m above
COM). The maintainer clarified torso turning, not large lateral bending.
The postured pickup path holds measured roll, uses the existing native body
height command for modest crouch, and then requests yaw/pitch through the
exclusive existing HAL waist relay. Native source limits and MC gains remain
unchanged. No post-start root/joint teleportation or artificial support is used.

The installed simulator configuration selects position control for the waist.
In episode013548Z, held waist-pitch actuator torque matched
40*(published_position-q)-5*qvel to within 0.0000012 Nm; the separately
published effort did not appear in the measured torque. The scoped simulation
adapter consequently encodes native gravity bias as a position setpoint
correction, bias/kp, with unchanged gains and original source effort/range
bounds. This is an installed-simulator observation, not a hardware contract.
See that episode's waist_position_mode_audit.json and the
[official joint interface](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/joint_control.html).

A torso bend can foreshorten an upright bottle in the image. The failed H220
frame had aspect1.225, below the old1.25 image rule despite valid depth.
The corrected proposal uses calibrated world-height support, then the existing
cylinder surface, top-plane and fit-quality checks. Saved-frame replay located
its centre within0.790 mm of evaluation-only truth; that is an offline
estimator result, not a live grasp result.

All these results retain the explicit SIMULATION_ONLY source-derived pad
collision/control profile, right-claw kp6/original2Nm limit, local pad contact
time constant0.002s, and the pre-existing FoundationWalk diagnostic overlay.
They are not vendor-default or hardware qualification.


**Authorization update, 2026-11-06:** the maintainer subsequently authorized
bounded gripper gain and joint-internal friction/damping diagnostics that
preserve the reviewed STL assembly and kinematics. See the owning
[model-integrity authorization](../engineering/model_integrity.md#c1-standing-authorization-issue-46-omnipicker-simulation-parameter-debugging).
The pending-approval statements below describe historical checkpoints and no
longer block the specified friction comparison. No experiment result is implied
by authorization; record the actual next run separately.

## Actual experiment

`reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T070115Z/`
contains the real native MC/simulator execution. Standing, docking and static
approved-circle IK passed. Arm execution completed, but TCP position error was
94.708 mm and closure was not issued. Bottle contact with R_hand_wide1_Link
was observed: this is not approved pad contact. The former pipeline did not
abort on that contact and did not verify jaw opening before approach.

At simulated time about 78 s, the requested opening was 0.50, actual
R_claw_joint was -0.09744 rad instead of -0.50 rad. Thus static clearance at
50% opening did not describe the actual approaching hand.

Offline FK comparison at the observed endpoint separates two effects:
actual arm angles with original captured root gave 2.591 mm position error;
actual arm angles with measured native root gave 96.866 mm. This supports a
stale/frozen-base planning defect, not proof that the arm IK solver is broken.
The later bottle pose also changed during forbidden contact; the numbers are
not a decomposition of every millimeter of final native error.

## Scoped repairs

- Parent owns exact-ABI native observer construction, injection and fresh readiness.
- Approved-circle adapter preserves ROS capture and uses a coherent native pelvis,
  leg, waist and arm sample for its next offline solve.
- Native arm feedback is explicitly SIMULATION_ONLY; ROS samples remain recorded.
- Missing, stale, regressing-generation native state aborts commands.
- Actual opening within 0.03 rad is required before approach.
- Non-pad bottle contact above 0.1 N aborts the arm path.
- Closure not executed is NOT_RUN, not a physical grasp FAIL.

These code repairs have focused offline tests. They have not established a
successful subsequent physical grasp, moving-base tracking, or lift.

## Opening torque evidence

`reviews/issue-46/truth-grasp-opening-forensics-20261106T100429Z/result.json`
records the unchanged compiled model, hashes, coupling derivatives and estimates.
Each of six right claw scalar joints inherits frictionloss=0.3 Nm. Quasi-static
virtual-work projection through the existing polynomial equalities estimates
1.823 Nm opposing motion at root=-0.5 rad. Existing kp=2, gear=-1,
ctrlrange=[0,1] permits only 1.0 Nm static servo effort there at ctrl=1.
At ctrl=0.5 the static position feedback effort tends to zero at the target.
This is a supported explanation for poor opening/tracking, not a measured
hardware torque specification or a complete solver proof.

The values are model defaults and an established experimental actuator overlay,
not verified OmniPicker hardware parameters. Changing them is not authorized by
this report. No friction, gain, force limit, geometry, topology or vendor policy
has been changed. Review the physical parameter provenance and approve an exact
separate actuator/friction delta before pursuing a modified-model grasp.

The earlier verbal suggestion kp=2 -> 10 alone is **not a validated repair**.
At q=-0.5, estimated balance command is 0.5 + 1.823/kp: about 1.411
with kp=2 (outside [0,1]) and 0.682 with kp=10. Merely increasing kp while
continuing to send command=0.5 still leaves Coulomb tracking error. At q=0,
estimated coupled resistance is 2.012 Nm, slightly above the unchanged 2 Nm
force cap. These estimates make an isolated controller/parameter review
necessary; they do not justify increasing force or silently reducing friction.
No kp change, feedforward-control trial, or new physics episode is approved
by an automatic Goal continuation. Keep the proposed gain change unimplemented.

Primary semantics: [MuJoCo 3.3.7 position actuator](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#actuator-position)
and [constraint/friction model](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html).

## Reproduction

From Windows PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec bash scripts/research/issue46/grasp/run_truth_circle_review.sh --pause-for-human-review
```

This is a full-robot native physics diagnostic, not a successful grasp demo.
The expected first blocker with unchanged parameters is actual opening. It
retains the failure window up to 15 minutes; normal bounded execution cleans up.
No new physical run is justified merely to rediscover the recorded opening
failure. Review the parameter correction first. The full objective remains
>=50 mm lift plus >=1 s unsupported hold and is not achieved.

## Startup fully-open request (2026-11-06)

User requested opening from initialization, holding open through walking, and
avoiding close/reopen on docking. Added explicit `--initial-hands-fully-open`
to the existing approved-circle adapter and native runner. The arm bootstrap
publishes both existing hand commands: delivered 0.8 s activation handshake,
then normalized 1.0 continuously. This changes command ordering only; it does
not write model state or alter actuator gains, force limits or geometry.
A fresh native opening check requires each root within 0.03 rad of -1 before
walking. The grasp child preserves an already-open activation state, then
checks the planned approach opening before moving arms.

Actual native run:
`reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T104944Z/`.
Reset and standing passed. At the pre-walk gate, L=-0.136957 and R=-0.137559
rad, target=-1.0 rad. Gate failed, so walking, approach, closure and lift were
NOT_RUN. Last cleanup-era sample at sim=12.101 s was about -0.250 rad with
ctrl=1.0 and actuator force about 1.499 Nm. This establishes actual early
opening progress, not full-opening success, permanent stall, or a new torque
measurement of hardware. Delayed full opening remains unproven; do not infer
it from the command or simply lengthen timeouts to claim success.

Focused offline suite including bootstrap generation, preservation of open
state, and pre-walk opening gate: 32 tests PASS. Container absence confirmed.

Reproduce this specific opening-first diagnostic with the previous command
plus `--initial-hands-fully-open`; do not present it as a successful grasp demo.

## Saved opening trace: command, coupling, and endpoint budget

`reviews/issue-46/startup-opening-linkage-analysis-20261106T105720Z/result.json`
analyzes all 1025 native samples from the startup-opening run, without new
physics. Native ctrl=1 confirms delivery to the simulator actuator, not just
Python publish success. The maximum residual across five right-hand follower
polynomials is 0.002225 rad. No hand/table or hand/bottle contact appears in
the captured contact scope. Internal hand self-contacts are not captured by
this observer, so their absence is not established.

At q=-1 the same virtual-work estimate is 2.7774 Nm, above the unchanged
2 Nm actuator force cap. Increasing kp alone cannot overcome that force cap.
The last root velocity remains -0.009873 rad/s: this recording does not prove
permanent stall or establish eventual full opening. Do not report that the
hand can never open based on this short trace.

The next mechanical/control review must address provenance and consistency
of the inherited 0.3 Nm friction on every added scalar joint versus the
experimental root actuator, rather than another arbitrary gain increase.
Model parameters remain unchanged; full-open walking and grasp remain unqualified.

## Friction provenance review and proposed causal comparison

The existing full-body MJCF default class x2 specifies joint frictionloss=0.3,
armature=0.03 and damping=0.0. The six right claw active scalar joint elements
lack explicit frictionloss and inherit this default. Checked official v1.4.0
OmniPicker URDF joints provide no matching dynamics values. Absence of a URDF
value is not evidence that real hardware friction is zero.

Exact source/scene hashes and a **not implemented** proposal are in
`reviews/issue-46/claw-friction-provenance-review-20261106T110223Z/`.

Preferred discriminating proposal supersedes gain-only tuning: one separate
SIMULATION_ONLY right-hand overlay setting frictionloss 0.3 -> 0 on only
R_claw_joint, R_hand_wide1_joint, R_hand_narrow3_joint,
R_hand_narrow_loop_joint, R_hand_wide3_joint, R_hand_wide_loop_joint.
Keep kp=2, kv=0.05, +/-2 Nm force, armature, all geometry/constraints and MC
unchanged. Run no-load opening for at most 8 seconds; no walking, approach or
bottle contact. This would test inherited friction as a cause, not establish
hardware parameters or authorize grasp acceptance. Case-specific human approval
is still required before producing that overlay or running it.

## Authorized friction isolation and live tracking implementation (2026-11-06)

The approved separate right-hand comparison is implemented in
`claw_parameter_diagnostic.py`; it changes only six explicit joint frictionloss
values to zero. Baseline and derived compiled numeric arrays are checked, with
exactly the selected loss entries excepted. A bilateral version then applies
the same correction to both hands. Separate host/runtime compilation parity
is checked. Geometry, joint/transmission/equality definitions, inertia, contact
friction and actuator gains/limits remain unchanged. This remains an assumption
for simulation diagnosis, not measured hardware friction.

Actual comparison:
`reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T113843Z/`.
Right opening after observation: -0.980390 rad, velocity 0.016782 rad/s;
left unchanged-friction reference: -0.293138 rad. Right opening gate PASS.
Bilateral startup opening subsequently passed in real native episodes.

The full-open approach now uses opening=100 and the reviewed match circle at
opening=40 for the 70mm object. It screens open, match and close-contact poses
and an approach sweep. Old 50%-open clearance checks are not reused to qualify
full-open motion.

Instantaneous native state is exposed via an atomic latest JSON sidecar while
all raw history remains in JSONL. The truth-diagnostic dock client can consume
fresh native pose and joints explicitly; ROS remains the command transport and
its raw telemetry is preserved. This is not LiDAR/odometry estimator acceptance.
The existing approved-circle IK can update its endpoint at 10Hz from measured
root/waist state, using MuJoCo point/axis analytic Jacobians. Missing/stale
updates, generation changes, failed IK, nonpad contact, or pre-close bottle
translation above 10mm/tilt above 0.10rad abort execution. Active freshness is
still 0.20s. Model-loading readiness precedes executable target creation.

Failure evidence remains retained:

- `...115106Z`: docking passed, arm initialization rejected stale native state.
- `...115755Z`: full-open arm approach used frozen-root endpoint, pushed bottle;
  closure NOT_RUN. This precedes the measured bottle-motion abort fix.
- `...120643Z`: tracking initialization did not converge; arm NOT_RUN.
- `...121439Z`: low neutral full-open hand contacted bottle before arm approach;
  tracker rejected bottle movement, closure NOT_RUN.
- `...122319Z`: attempted 0.8rad shoulder transit park disrupted route heading;
  position arrival failed. This park is optional and not the default.
- `...123355Z`: safer 0.74m docking target arrived; tracking startup state stale;
  arm NOT_RUN. Added fresh pre-target readiness, not a relaxed active deadline.
- `...123730Z`: command-subscriber discovery race before Reset; readiness now
  requires exactly one discovered arm subscriber.

Above suffixes refer to full directories beginning
`reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T`.
Do not label any of these episodes a physical grasp PASS.

The current default truth docking x target is 0.74 instead of 0.89m, with y
unchanged -1.44m, to keep the low full-open transit hand away from the bottle.
An offline representative stance x=0.85m passed open/contact/sweep checks at
`reviews/issue-46/approved-circle-standing-ik-20261106T123153Z/`.
It is not a live arrival/grasp guarantee or a change to bottle/table geometry.

Full reproduction (currently a diagnostic, not a grasp-success demo):

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec bash scripts/research/issue46/grasp/run_truth_circle_review.sh --initial-hands-fully-open --both-claws-zero-internal-friction --dock-world-xy 0.74 -1.44 --pause-for-human-review
```

On a manipulation failure the runner attempts process/window-verified native
pause and verifies stopped simulation time before retaining the renderer for
15-minute multi-angle review. Do not resume physics manually; use the recorded
release marker for teardown. Earlier-stage failures still clean up immediately.
No lift or >=1s unsupported hold has been qualified.

### Follow-up endpoint and timing corrections

The default truth diagnostic docking x target was refined from 0.74 to 0.80m:
the actual 0.74m-target run stopped at x=0.797m and static IK failed its wrist
margin; the same captured pose translated +53mm passed all static gates at
`approved-circle-standing-ik-20261106T124527Z`. The 0.8rad shoulder park remains
optional and disabled by default, because its live route did not arrive.

`...124738Z` reached manipulation startup, but the iterative tracker exceeded
its 0.20s compute/feedback deadline before approach. Tracking FK now uses only
mj_kinematics and mj_comPos for the analytic point/axis Jacobian, with damped
least-squares updates and bound-aware recovery; full preflight collision scans
still use mj_forward. A 30-frame saved-state replay measured max 3.1ms and
<0.1mm centre residual; this is offline timing, not a runtime guarantee.

`...125429Z` rejected initial tracking near the wrist limit. Fixed an
inconsistent 0.01rad tracking margin by aligning it to the original static
0.001rad source-limit margin. No physical source limits changed. Same-state
offline checks at section offsets 0.05/0.10m found the higher 0.10m section
with 0.086rad minimum margin; the lower one did not converge. The 0.10m
section leaves 20mm nominal top margin for the canonical 300mm bottle and
30mm axial half-width pad. It still requires actual approach/contact proof.

Native review-pause verification now waits up to 3s for 0.5s unchanged native
simulation time after one verified space event, rather than presuming the
renderer processes input within a fixed 0.35s sleep. A dispatched key alone
never establishes paused physics. Preserve any original execution failure
independently if pause verification fails.

### Current human-review checkpoint

`reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T130347Z/`:
actual bilateral startup opening, native-feedback table docking/stop and static
IK passed. Actual arm approach executed. First measured circle error was
55.345mm, axis error 0.136171rad. After residual compensation, actual centre
error was 4.245mm and axis error 0.008740rad: native arrival PASS. The trace
then contains 19 HAND_CLOSE_HOLD samples over about 1.1s, and actual native
actuator ctrl changed from 1.0 to 0.39 with joint movement. The tracker aborted
because its pre-close 10mm bottle-displacement test incorrectly remained active
during closure. Earlier parent/checkpoint NOT_RUN/no-closure labels are
superseded by closure_stage_correction.json; no stable bilateral grasp PASS
is established by this partial close.
Native physics pause was confirmed at sim=93.581s; screenshot and
`human_review_checkpoint.json` identify the full robot/table/bottle review.
The bounded 15-minute retained renderer is a human-review checkpoint, not
successful grasp or lift. Do not automatically rerun before reviewing this
endpoint/contact geometry. Goal remains incomplete.

Reproduce that exact reviewed section with the full command above, changing
`--dock-world-xy 0.80 -1.44` and adding
`--grasp-height-above-center-m 0.10`. Frozen vendor assets, gains, force limits,
contact friction and model geometry are unchanged; zero joint-internal loss
remains a separately labeled SIMULATION_ONLY assumption.

Closure-stage reporting now distinguishes publish start from completed hold:
the child emits close publish-call count even on exceptions, and the parent
parses it on nonzero child exit. Python publication remains distinct from
native actuator receipt; this particular run has separate native ctrl evidence.
The next scoped correction is phase-aware tracker observation during closure,
plus damping review for the measured root oscillation. Human review checkpoint
remains active before another trial; no automatic new attempt is authorized
by the checkpoint itself.

### Offline phase correction pending review (no new physical episode)

After the retained human-review run, tracking now accepts an explicit CLOSING
handoff only after native arrival PASS and matching reset generation. The
closure target is frozen at the attained world anchor instead of chasing the
object. Pre-close and during-close motion faults are reported separately.
The same 10mm translation and 0.10rad tilt safeguards remain active; this is
not permission to accept a bottle pushed out of the grasp. Phase expiry is
bounded and native telemetry freshness remains 0.20s.

42 focused offline tests PASS, including frozen-anchor behavior, >10mm motion
rejection in both phases, false-arrival and generation-mismatch rejection.
This correction has not been physically validated; do not launch a new episode
until the actual lower-pad placement at the saved review checkpoint is reviewed.

### Recorded native-state review, without another physical attempt

The existing approved-circle entry point now accepts `--review-captured-run`.
It verifies the exact derived scene/replan hashes and restores all recorded
scalar joints plus pelvis and bottle poses in a separate display MjData.
It does not call mj_step, Docker, ROS or MC. GUI keys 1/2/3 select attained
arrival, partial closing and final captured state; arrows change frames and
space plays recorded frames. Yellow shows the actual hand-frame support circle;
magenta marks the measured bottle target section. Full robot and scene remain
visible. It is clearly labeled RECORDED REVIEW / NO PHYSICS.

The capture used native 3.3.7; the display uses existing Python MuJoCo 3.3.6.
Representative recorded/rendered body translations must agree within 5mm;
static display contacts/forces cannot be promoted to new physical evidence.
This review avoids rerunning motion merely because the retained live window
expired. The prior native physical result and unresolved grasp/lift gates
remain unchanged.

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec bash scripts/research/issue46/grasp/run_truth_circle_review.sh --review-captured-run /home/lzy18001500226/projects/RobotSim/reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T130347Z
```

### Maintainer visual acceptance of attained posture (2026-11-06)

The maintainer reviewed the recorded full-robot native arrival pose
(sim=91.021s, centre error 4.245mm) from the 20261106T130347Z episode and
confirmed the visual placement is acceptable to proceed with actual closure.
This resolves the human-review checkpoint; it does not establish bilateral
force-bearing grasp, full closure hold, lift, or hardware validity. The next
bounded native episode may validate the already tested phase-aware closure
path. Ordinary authorized parameter diagnostics retain their standing scope.

### Physical closure and lift recovery (2026-11-07 local date)

The five-stage autonomous scope is recorded in the owning
[task workflow](../workflows/06_X2_AUTONOMY_TASK.md). Stage 1 remains the priority;
later perception/localization capabilities cannot be inferred from it.

The following actual native runs share the prefix
`reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T` (UTC names).
Every listed terminated container was positively confirmed absent.

| Suffix | Actual decisive observation |
| --- | --- |
| `163312Z` | Full startup/opening/standing PASS; docking stopped outside 100mm tolerance. Arm/closure/lift NOT RUN. |
| `163723Z` | Key remained -1, so Load key restored the ordinary start rather than captured stance. No grasp. |
| `164807Z` | Native key-slider auto-load PASS; full opening and approach executed; 5.508mm arrival exceeded the unchanged 5mm gate. |
| `165524Z` | Loading the almost-touching attained pose let initialization knock the bottle off the table. Replaced this episode-boundary input with the same capture's docked **pre-approach** pose. |
| `171134Z` | Direct seven-joint error compensation demanded a shoulder-roll limit crossing. Replaced it with bounded measured Cartesian/axis correction, not a source-limit change. |
| `171640Z` | 4.351mm actual arrival PASS, closure executed, but contact repeatedly rebounded. Root q oscillated approximately -0.376 to -0.577rad. |
| `172653Z` | Separate right-claw kv=1.27 diagnostic: 4.513mm arrival and 1.50 simulated seconds of bilateral contact PASS; minimum pad normals 0.410/0.356N. No lift yet. |
| `173643Z` | 39% static contact-probe command did not transfer load: 8mm commanded probe produced only 0.163mm bottle rise; table normal remained 1.335N. |
| `174122Z` | Full closed target produced sufficient force but pushed the bottle beyond the existing 10mm closure-motion guard; aborted. |
| `175103Z` | Force-guided closure was stable; a lift observer incorrectly charged pre-close movement against the new lift baseline. Retained failure; corrected stage baseline. |
| `175855Z` | Actual hand rose 2.151mm during the one-second probe and was still moving. Bottle rose 0.385mm; table retained 1.064N. No load-transfer PASS. |
| `180821Z` | Arrival refined to 0.942mm; bounded measured lift feedback ran. Actual hand rose 10.847mm but bottle rose only 1.094mm; measured hand-relative translation was 11.168mm. Genuine slip guard FAIL, not a publisher/IK success. |

The current derived scene's canonical blue cylinder is **70mm diameter,
300mm high, 0.200kg**, not a qualified 570g payload. Do not transfer a future
PASS to another mass. The original source model, table, object mass/inertia,
contact geometry, friction, condim and actuator force caps remain unchanged.

The right-claw damping comparison changes kv from 0.05 to 1.27 only; native
kp remains 2 and force range remains +/-2Nm. The local constrained-tangent
inertia estimate is about 0.2027 kg m2, giving critical kv about 1.2735.
The earlier static planning model has kp=0.5; that must not be mistaken for
the actual runtime actuator. This is simulation control calibration, not
hardware identification. Exact parameter deltas, all compiled-array checks,
host/runtime parity and hashes are in each overlay result.

Truth planning now uses the same native bottle/base/joint snapshot. XML default
object coordinates cannot stand in for this episode's measured object pose.
Physical lift is separately gated by a zero-step 13-sample collision/limit sweep,
sustained actual pad contact, an 8mm clearance/load-transfer probe, then a 60mm
bounded hand trajectory. Independent acceptance requires >=50mm **object** rise,
>=1s continuous unsupported hold, no table/other support, and bilateral forces.
Lift slip is relative to the measured hand transform, separately from servo
tracking error. No object state is written during execution.

At the failed 180821Z snapshot there are two active distal-pad contacts with
condim=3. A Coulomb point-contact wrench check cannot balance gravity at that
pose (gravity torque about their contact line: 0.01717Nm). This is a local
snapshot diagnosis, not a proof that every grasp pose is impossible. Two bounded
zero-step centre/opening solves did not produce a qualified alternate frame.
The reviewed STL circle remains the nominal planning input, not a physical
force-closure guarantee.

Exact-engine 3.3.7 offline `multiccd` comparison added table contacts but no pad
contacts; it was not promoted to a live change. The official
[MuJoCo slip guidance](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip)
also identifies soft-friction drift and the numerical Noslip postsolver.
A separately flagged `--noslip-solver-diagnostic` selects exactly two numerical
postsolver iterations, with identical geometry, contact friction/condim and
force limits. This is explicitly not vendor-default physics; the documentation
notes that Noslip makes inverse dynamics ill-defined. Its actual physical
qualification remains separate from implementation and static checks.

Fixed-stance reproduction (no walking claim):

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec bash scripts/research/issue46/grasp/run_truth_circle_review.sh --initial-hands-fully-open --both-claws-zero-internal-friction --right-claw-damping 1.27 --noslip-solver-diagnostic --fixed-reviewed-checkpoint /home/lzy18001500226/projects/RobotSim/reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T130347Z --grasp-height-above-center-m 0.10 --physical-lift-test
```

That Noslip-only configuration failed and is not the working reproduction.
Full start-to-table execution and fresh RGB-D substitution are separate
acceptance gates. All tooling remains research-only; no production promotion
or model merge is authorized by these experiments.

## First measured full-flow lift (2026-11-07)

The source-preserving whole-mesh collision baseline failed. At the same approved
40% circle pose (radius 35.07825mm), its convexified narrow/wide distal meshes
intrude 2.38859/2.38696mm into the circle. Source/native mesh vertices agree to
3.4e-16m. See `native-contact-envelope-audit-20261106T194437Z/result.json` and
its `source_vs_convex.png`, plus `native-pad-collision-mesh-parity-20261106T194717Z`.
This is an envelope diagnosis, not an excuse to suppress collision contacts.

The separately named `--source-pad-coacd-comparison` reuses exactly 12 narrow3
and 10 wide3 parts from the existing locked `collision_proxy_manifest.json`
(SHA256 `7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a`).
Every original STL and part hash is verified. Only those two collision geoms
are replaced in a generated variant. Visuals, joints, inertias, force limits,
contact friction/condim, exclusions, table and object are unchanged. The old
CoACD approximation has a 3mm metric target and part-count cap; it is not exact
vendor geometry. Measured circle intrusion falls to 0.50940/0.56011mm.

| Actual native 3.3.7 run | Fixed-stance 203002Z | Full START-to-table 210324Z |
| --- | ---: | ---: |
| Reset / standing / approach / closure | PASS | PASS |
| Walk / final dock error | NOT RUN | PASS / 0.010602m |
| Minimum bottle rise in accepted hold | 0.058488m | 0.055597m |
| Native hold samples / span | 51 / 1.000s | 51 / 1.000s |
| Minimum narrow / wide normal force | 1.67154 / 1.65129N | 1.75457 / 1.73661N |
| Mean upward pad force / object weight | 1.96162 / 1.962N | 1.96182 / 1.962N |
| Table / other support | zero | zero |
| Container positively absent after teardown | YES | YES |

Both outputs are under canonical `reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T...`.
The 0.2kg free bottle has no actuator, weld, attachment, support trick or active
object-pose write. Force-guided closure remains within the original +/-2Nm
actuator cap. The right claw uses kp=2 unchanged, kv=1.27; both claws use zero
internal joint friction. The passing solver selects elliptic cone/impratio=10,
Noslip=0. These are explicit simulation assumptions, not a vendor baseline PASS.

All 51 accepted full-flow frames were independently mapped back to original
source triangles in `native-lift-pad-region-audit-20261106T211301Z/result.json`:
two contacts per frame, both within the human-approved lower concave arcs,
off-region force zero, maximum contact-to-source-surface distance 0.38714mm.
This does not establish four-point contact or indefinite carrying stability.

The intervening fixed-stance confirmation 203535Z stopped before closure after
a >0.20s native-observer freshness gap. No failed freshness gate was relaxed.
The successful full-flow run retained native wall/step/I/O timing measurements.

Independent full-flow repeat `20261106T212727Z` passed every executed gate:
51 terminal samples over 1.000s, minimum rise 57.09596mm, minimum pad normals
1.67934/1.65946N, mean upward force 1.96142N, table force zero, final position
range 2.348/1.774/2.365mm and positively confirmed container removal. This
qualifies repeatability for this explicit truth/proxy profile, not other
stances, bottle heights, fresh perception or hardware.

Reproduce the actual full flow (Linux arguments are passed directly by WSL,
avoiding nested PowerShell/Bash command substitution):

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec env DISPLAY=:0 MUJOCO_GL=glfw bash scripts/research/issue46/grasp/run_truth_circle_review.sh --source-pad-coacd-comparison --initial-hands-fully-open --both-claws-zero-internal-friction --right-claw-damping 1.27 --elliptic-grip-solver-diagnostic --dock-world-xy 0.85 -1.49 --dock-braking-delay-s 0.4130254699850907 --grasp-height-above-center-m 0.10 --physical-lift-test --pause-for-human-review
```

The optional review flag pauses native physics only after the lift result has
been measured, leaving camera/window controls interactive for up to 15 minutes.
Review time cannot count toward the hold. Ctrl+C triggers container cleanup.
Recorded full-flow review needs no new Docker/ROS/physics:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec env DISPLAY=:0 MUJOCO_GL=glfw bash scripts/research/issue46/grasp/run_truth_circle_review.sh --review-captured-run /home/lzy18001500226/projects/RobotSim/reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261106T210324Z --review-full-flow
```

Replay stops at the qualified controlled terminal frame; key 4 separately shows
the preserved post-control cleanup tail. Renderer 3.3.6/native 3.3.7 body pose
parity is 0.00188mm. Replay is labelled NO PHYSICS and is not a new validation.

## Bounded stance and RGB-D continuation

`approved-circle-bounded-stance-region-20261106T214002Z` tests seven specified
nearby x/y/yaw samples for each of the separately identified 300/150mm bottle
heights. These are static samples, not a continuously or dynamically qualified
region. With the initial numerical lift screen, six 300mm samples passed.
The 150mm straight-waist candidates hit the unchanged right shoulder-roll
upper limit 0.061rad and wrist-pitch lower limit -0.558rad, leaving 9--17mm
position and 0.21--0.27rad axis errors. Supported waist/posture investigation
is still required; the 300mm PASS does not qualify the smaller height.

The first +20mm-x physical attempt 214927Z stopped before arm execution:
open/closed pose and approach sweep passed, but two finite-difference lift
samples exceeded the unchanged 0.001 axis-residual gate. The exact MuJoCo
point/axis Jacobian matches a central-difference regression and resolves that
saved pose without changing tolerances: maximum residuals 0.322mm and 0.000765.
The corrected physical trial 220213Z then passed: measured initial root
(0.895066,-1.527933,0.650242)m, minimum lift 56.986mm, 51 frames/1.000s,
minimum normals 1.66424/1.64508N, upward force 1.96138N, no table support,
cleanup confirmed. This is a fixed-stance offset test, not a navigation claim.

The fresh RGB-D adapter is a **calibrated virtual sensor** at the source head
mount position with an explicitly chosen fixed 0.65rad downward optical view.
It does not point at an object truth pose, move the head in the live model,
use a truth mask, or claim manufacturer optical calibration. RGB/depth use the
same recorded/live native configuration and camera transform. The estimator
reuses the task-blue 4-connected component rule, measures variable radius from
side depth, and detects a horizontal top band. It assumes an upright blue
cylinder and an explicit 300/150mm height prior. It is not generic 6D object
recognition or a hardware depth-noise qualification.

Saved-pose preflight 214510Z gave 0.333mm XY, 0.315mm diameter and 0.338mm top
errors at 640x480. At 320x240, `rgbd-visual-fk-resolution-preflight-20261106T221954Z`
gave 0.654/0.361/0.651mm respectively. Visual-only FK produced exactly the
same RGB and depth as full `mj_forward` for that pose, avoiding an unnecessary
second contact solve. Sensor rendering disables aesthetic shadows/reflections;
native physics and the human-view scene are unchanged.

Early live RGB-D attempts stopped before the arm moved. 220648Z exposed a
first-frame render initialization deadline; warmup is now excluded from control
feedback. 221452Z/222304Z exposed the isolated IK environment's missing Pillow
through an unnecessarily heavy contract import. The data validator now imports
without image dependencies in the exact uv IK environment. Subsequent saved-pose
and live traces exposed periodic render stalls; native physics continued with
fresh 20ms samples. Never infer that an MC failure caused a host sensor stall.

Offline co-load benchmarks (no simulator/controller/motion) reproduced these
stalls with a static saved pose, including after the IK child exited. Limiting
BLAS threads or disabling cyclic GC did not remove them. Default WSLg renderer
was D3D12 Intel Iris Xe: maxima 0.55s. Explicit D3D12 RTX 4060 selection produced
100 frames with maximum render/fit 0.1204s and no >0.15s frames in
`rgbd-offline-coload-benchmark-20261106T225230Z`. This supports a renderer-path
diagnosis, not a claim of every driver-internal cause. Use the documented
[WSLg GPU selection](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg)
only in the sensor process; verify the actual GL renderer string. No driver,
system-wide GPU setting, controller or safety deadline is changed.

The host model/camera cache is compiled before native stepping and binds source
XML, camera code, MJB and exact 3.3.6 engine hashes. Body/joint/geom/contact/
actuator/equality/initial-state arrays are compared before caching. The native
3.3.7 simulator still loads the original selected XML. Active pose and image
freshness remain 0.20s, with generation matching and no duplicate age refresh.
At verified arrival the perception target freezes; contact/proprioceptive
feedback owns closure/lift, with independent truth used only for safety/audit.
Fresh live RGB-D grasp acceptance remains separate from these offline results.

The RTX-selected live trial `20261106T225655Z` passed static RGB-D IK and began
actual arm approach. Its 418 qualified images span about 33.7s; the actual
renderer was D3D12 NVIDIA GeForce RTX 4060 Laptop GPU. Before closure, one
foreground/silhouette depth contamination made the algebraic circle initialize
at the 80mm radius bound, with 4.10mm residual. The image-fit gate stopped the
arm and sent no close command; cleanup was confirmed. Native bottle tilt was
only 0.00011--0.00026rad in the surrounding eight frames, so a falling bottle
does not explain this fit error.

`rgbd-failed-approach-frame-replay-20261106T230344Z` reproduces five bad frames
from those native poses without new physics and retains RGB/depth/calibration.
One-pixel RGB-mask boundary exclusion plus 32 bounded deterministic circle
hypotheses and >=80% current-depth inliers resolves all five in
`rgbd-occlusion-robust-fit-replay-20261106T230658Z`: XY errors 0.638--0.655mm,
radius approximately 34.82mm, inlier fraction 1.0. This is confidence filtering
of sensor pixels, not a change to a gripper contact surface or physics. The
old residual/angular-support gates remain. The regression corrupts silhouette
depth while preserving blue RGB and verifies that the fitted centre cannot
jump onto a foreground hand. A new live test is still required.

For the 150mm case, +0.12/+0.24rad waist-pitch screens and a separately tested
closer/shoulder-aligned stance did not qualify; the explicit opposite cylindrical
axis branch also failed its source-limit/clearance checks. No such failed
candidate was sent to the physical controller. These local bounded failures
are not proof of global IK impossibility. In particular, the existing 1.1rad
arm-transition excursion bound and the source wrist/shoulder limits remain;
unproven larger-arm or lower-body motions are not silently enabled.

## Canonical fresh RGB-D physical acceptance

The continuous-vision fixed-stance run `20261106T234215Z` passed after target
handoff at the first actual qualified arrival: position error 3.583mm and axis
error 0.01307rad. This freezes the RGB-D estimate for bounded proprioceptive
refinement, not automatic closure. The original 5mm/0.03rad arrival, joint,
posture, contact and lift gates remain. There were 571 qualified images,
maximum processing age 0.1578s, and one delayed image discarded without
refreshing its stamp. Native feedback remains limited to 0.20s. The actual
lift minimum was 57.820mm over 51 samples/1.000s, with minimum normals
1.65199/1.63090N, upward force 1.96175N and no table support.

Full continuous-vision run `20261106T234557Z` stopped before closure when the
hand occluded the bottle earlier from its different attained docking pose.
It is preserved as FAIL/NOT RUN for closure/lift. Perception cannot see through
the hand; confidence failure is not converted into a successful estimate.

The distinct `fresh-rgbd-snapshot` mode uses a freshly qualified planning image,
then a second independently stamped image before arm execution. Position and
diameter must agree within 2mm. The bounded world target is explicitly static,
expires after 45s, and invalidates on Reset generation change or altered data.
It never reports the old image as a new live frame. Encoder/base feedback,
actual TCP arrival, object-motion/fall/contact guards, closing force and free-body
lift acceptance remain unchanged. This reuses the staged object-pose/IK/approach
idea in the official [MTC pick example](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html);
its planning-scene attachment and broad collision allowances are **not** copied
into MuJoCo physics. No new navigation framework or vendor change is introduced.

Full snapshot run `20261107T000115Z` passed START -> actual walking -> table
stop -> RGB-D -> circle IK -> measured arrival -> bilateral closure -> lift:

| Measurement | Actual native result |
| --- | ---: |
| Start / dock XY | (-1.490295,-1.509751) / (0.903756,-1.525896)m |
| Nominal dock target error | 64.640mm (unchanged 100mm gate) |
| Fresh reobservation difference | 0.09262mm |
| RGB-D estimated diameter | 69.731mm |
| Minimum accepted lift / hold | 57.521mm / 51 frames, 1.000s |
| Minimum narrow / wide pad normal | 1.67184 / 1.65989N |
| Mean upward pad force | 1.96187N (object weight 1.962N) |
| Table support / cleanup | zero / container positively absent |

The current sensor assumes a known cylinder height and an upright axis; this
is not generic unknown-height/6D recognition or hardware camera calibration.
The 64.6mm dock error did not prevent this actual grasp because IK replanned
from the attained base state. It does not establish every intervening stance.

Exact-engine source-face audits `native-lift-pad-region-audit-20261107T000651Z`
and its `-01` sibling passed all 51 held frames for the fixed/full RGB-D runs.
Maximum proxy-point-to-source-surface distances were 0.417/0.740mm; region-outside
force was zero. The full run had 3--4 proxy solver contacts per frame; this
does not prove four independent physical contact bands or real contact area.

Full RGB-D reproduction, with native physics paused only after the measured
result for multi-angle human review:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec env DISPLAY=:0 MUJOCO_GL=glfw bash scripts/research/issue46/grasp/run_truth_circle_review.sh --source-pad-coacd-comparison --initial-hands-fully-open --both-claws-zero-internal-friction --right-claw-damping 1.27 --elliptic-grip-solver-diagnostic --target-source fresh-rgbd-snapshot --rgbd-render-adapter NVIDIA --native-render-adapter NVIDIA --dock-world-xy 0.85 -1.49 --dock-braking-delay-s 0.4130254699850907 --grasp-height-above-center-m 0.10 --physical-lift-test --pause-for-human-review
```

Recorded full-flow review uses the same wrapper with
`--review-captured-run /home/lzy18001500226/projects/RobotSim/reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261107T000115Z --review-full-flow`.
It replays actual recorded states, does not rerun physics, and permits free
camera rotation. Source-region audit reproduction (zero physics):

```bash
/home/lzy18001500226/.local/bin/uv run --offline --with mujoco==3.3.7 --with numpy==2.2.6 --with scipy==1.15.3 --with trimesh==4.7.4 python3 scripts/research/issue46/native_runtime/native_lift_contact_region_audit.py --capture /home/lzy18001500226/projects/RobotSim/reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261107T000115Z
```

### Native-window graphics repair and independent review-path confirmation

Repeat 002336Z stopped before walking after a real 4.598s native sample gap:
`mj_step` calls were separated by 4.578s while individual physics steps were
below 0.73ms. This was not classified as a physical standing/grasp failure.
The actual native GL context then proved the old container was using llvmpipe;
the harness explicitly forced `LIBGL_ALWAYS_SOFTWARE=1` and a llvmpipe loader.
An adapter-name request alone could not enable hardware rendering.

`wslg-native-gpu-no-physics-smoke-20261107T004910Z` verified RTX4060 inside the
same unprivileged image without MuJoCo, MC, ROS or any motion: only `/dev/dxg`,
the existing X socket and read-only `/usr/lib/wsl` driver libraries were added.
There was no installation, privileged container, host network/IPC, global GPU
setting or host modification. Cleanup was positively verified.

The optional `--native-render-adapter NVIDIA` keeps the old headless software
path unchanged and selects hardware only for the native simulator process.
The observer reads `glGetString` after the genuine `mjr_makeContext`, calls
the original function exactly once, and never writes model/control/GL state.
The actual GL PID must match the native physics PID and renderer name before
MC startup. Environment-variable presence is not accepted as device proof.

Independent full RGB-D repeat 005238Z passed with the actual native context
`D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)`: dock error 30.639mm, minimum lift
56.599mm over 51 frames/1.000s, minimum normals 1.63915/1.62595N, upward force
1.96217N, no table support. The maximum observed post-Reset active frame gap
was 0.11144s over 4332 frames, retaining the 0.20s gate. This is an observed
run, not a hard real-time guarantee.

The native review pause also passed: physics stopped at sim86.861s, its actual
PID30 matched the visible 1706x1066 window and GL context, and camera/window
inspection remained available. The agent released this **automated pause
self-test**, not a human approval; container removal was confirmed. Tracking
stops before review-only pause so a deliberately frozen clock cannot create
a misleading stale-control error. Review time never counts toward lift hold.


### 2026-11-07 矮瓶边界与小步诊断准备

300 mm 瓶的真值、固定站位 RGB-D、从 START 行走后的 RGB-D 实际抬升已经通过；不能外推为任意瓶高通过。
150 mm 瓶保留同样直径与质量、底面不变，离线检查现有固定腿/根位姿下的腰 yaw/pitch 与七轴手臂。源范围诊断明确禁止 runtime handoff。最高受检截面距瓶顶 10 mm 时仍有 2.60 mm 位置误差、0.09127 rad 轴误差，原静态门未通过；没有执行该失败姿态。证据：reviews/issue-46/150mm-source-upper-top-section-20261107T020347Z/。这说明受检姿态不合格，不是全空间不可能的证明。没有修改源关节限位、腰/腿控制器或已审核包持圆。

后续站位小步复用现有 Dock、官方 cmd_vel_bridge 和历史 70a84cd 物理停稳计算。单次 0.2 m/s、0.4–1.0 s、零 yaw 的体坐标脉冲；桥独立 cmd timeout 0.20 s。必须先通过原 composite stop，脉冲后零命令至少 1.05 s，再一次 STAND 请求及完整合格停稳窗口；不通过不重启下一次运动。内部定时器仍是同进程，不能代替独立桥超时或物理停稳。MuJoCo 3.3.7 的 XBODY 原点世界速度已通过零步 API 对照测试，区别于 BODY 的惯性质心速度。新微步能力尚待实际动力学验证。


### 新增原生腿速审计：抓取证据与行走限速资格分别报告

微步 episode 20261107T033845Z 已通过原 composite pre-stop、精确 MC PID/节点/QoS 和实际桥输出观察；0.2 m/s 左移启动后约 0.179 s，原生右膝速度 11.941105 rad/s，超过固定源 URDF 11.93808，触发零命令并禁止继续。实际安装 MC robot_model.yaml 的同关节上限为 11.936 rad/s；后续检查取两源较小值，未提高限值。实际输入界限 McCommonState 为 forward [-1,0.8]、lateral [-1,1]，没有暴露 0.2 最低非零值；官方 bridge 的 min_* 是独立参数。

对既有两次 RGB-D 完整流程补做相同行走区间审计：005238Z 右/左膝峰值为 13.095641/13.669924 rad/s；000115Z 为 13.346165/13.946299 rad/s。这些历史运行的抓取承力、无支撑抬升测量仍有效，但当时父 runner 没有原生腿速逐帧门，不能继续描述为已通过此次新增源腿速全链路资格。保留原始记录，不改写历史结果。

下一项不同假设为外部 bridge 的较小 0.12 m/s 输入诊断：保留原 MC、FoundationWalk 配置、模型、力和关节限值，只使用 bridge 已公开的 min_forward/min_lateral 参数防止把小请求强制抬到 0.2。输入先与现场 MC 报告的 bounds 核对。仍为单次 0.6 s、严格停止和零重启；不是 MC policy 调参，也不能先验宣称合格。


微步 035847Z：0.12 m/s、1.0 s 左移输入，在源限值检查下得到停稳后 forward +0.040386 m、left +0.024546 m、yaw -0.017579 rad；Composite stop 测量 1.984319 s / 100 样本，目标最大变化 0.018598 rad，PASS。该检查按当时 50 Hz 保存帧进行；后续 C observer 已增加每个真实 physics step 的关节速度区间峰值，防止采样间漏峰。

完整较低线速度流程 040934Z 在到桌前之前安全中止，不能算抓取失败：输入 forward 0.12 / 初始 yaw -0.159021，后续小 yaw 被官方桥提升到最小 0.1；约启动 0.591 s 时实际 pelvis 原点速度 0.220098 m/s，COM 0.220532 m/s，超过 0.20。两种速度定义都越界，不能归因于原点/质心的语义差异。没有执行机械臂、闭合或抬升。

后续外部输入配置使用原已验证的更小 angular cap 0.075 rad/s，并同步约束 bridge 的 min/max angular，避免低值请求被抬高。MC 策略、增益和物理限值不变。物理故障保留首次 source timestamp，不让后续零命令观察覆盖首个原因。


### 150 mm 矮瓶：现有腰侧倾自由度的有界几何修正

此前 source-upper 搜索冻结了既有 waist_roll_joint，并非制造商关节缺失。依照当前 Goal 对弯腰/侧身的要求，复用同一 IK，增加这一现有源坐标，仍冻结 root/legs。没有改 joint、axis、限位、网格、MC 或碰撞筛选。三种有限种子均得到合格终点。首个目标 waist yaw/pitch/roll = (0.034768, 0.286095, 0.161869) rad。

证据 150mm-existing-waist-roll-diagnostic-20261107T050007Z：三个种子都通过 21 点张开接近扫掠、13 点60mm抬升几何扫掠、双侧 pad 接触、桌面/非pad阻挡及相对 baseline 的新增活跃自碰撞检查。沿扫掠的静态 COM 到原始双足接触凸包最小余量为39.66–40.80mm。几何中心残差约2e-9m、轴残差约1e-8rad只是求解精度，绝不代表感知或真实执行精度。

这仍是 ZERO_STEP，不是动态平衡、物理抓取或 MC 腰部接管 PASS。现有两度普通姿态偏好不能当作源机械限位；300mm已验证姿态保持原样，短瓶候选单独标记。接下来的受支持腰部接管和动态验证必须保留全部原安全门。

复现（只进行静态诊断）：
env OPENBLAS_NUM_THREADS=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46/grasp/approved_circle_stance_region.py --source-roll-diagnostic /home/lzy18001500226/projects/RobotSim/reviews/issue-46/approved-circle-standing-ik-20261107T020355Z/result.json


### 原厂全身策略与抓取占用的边界检查（尚未 live）

安装 foundationwalk_config.yaml 明确定义 action_nums=29，action_seq 包含14个双臂关节。现有抓取集成从启动就关闭MC arm publisher，并由 bootstrap 固定在(.4,0,0,-1.2,...)；这不等于完整保留原厂行走时的29轴输出。该差异是可验证的集成假设，尚未证明就是根速度越界的原因；还需观察实际MC臂命令，不能只凭配置存在断言其已应用。

准备复用已存在的腰部 topic relay 机制，将原生MC arm输出原样接入导航，站稳后再唯一接管抓取。当前只新增了独立配置/worker构建接口和离线消息原样转发测试，尚未接入父runner或执行新物理。禁止把此准备状态当成行走修复 PASS。

只读gain capture亦修正为先取得新鲜完整 leg/waist，再测满2s窗口：discovery最多2s，总计最多4s。重复源时间戳不刷新计数/年龄，中途陈旧立即失败，源时间跨度至少1.90s。不是用更长超时掩盖丢失数据。


### 2026-11-07 实时阶段3审核及 WSLg 输入恢复

用户明确要求实时物理，不能继续用记录回放代替。新增持久入口：
wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec env DISPLAY=:0 bash scripts/research/issue46/grasp/run_live_rgbd_grasp_review.sh

仅从已记录且碰撞检查过的桌前 episode-boundary 原生 keyframe 起步，30cm/D70/0.200kg，自由瓶、实际 native mj_step + MC、新鲜RGB-D抓前重观测和有界snapshot目标；不包含行走，也不包含15cm瓶。保持既有源派生 CoACD/控制诊断分类。没有使用回放替代物理。结束后通过原生GUI暂停物理、保留可交互窗口，关闭或终端Ctrl+C后清理。窗口自动审核等待上限900s；禁止恢复已结束控制阶段的物理。

首次063343Z在任何keyframe输入前被焦点检查拒绝（0个鼠标按钮事件），容器已清理。此次观测到Windows物理鼠标键全松开、X11 mask1040仍含Button3按下。XTest release和Windows release-only尝试未消除，未假称恢复。确认无其他Linux GUI/实验容器后，只重启受WSLGd监管的Weston图形进程，没有关闭Ubuntu或Docker。恢复后X11按钮位清零，Docker仍为29.8.1。证据 wslg-stale-input-recovery-20261107T071644Z。

入口现加入只读Windows/X11鼠标状态前检；物理按着时要求先松开，无法确认或持续不一致时不启动仿真；不自动重启用户的其他图形程序。另在可见物理审核的原生初始化和暂停前，精确匹配Windows标题、WSLg进程路径与唯一窗口并验证前台激活，之后仍保留原X11 PID/XID/title/focus及实际原生状态代次检查。未放宽输入验证或机器人门。

072640Z实际实时RGB-D抓取 PASS：最小抬升60.524283mm，最后51个样本/1.000s无支撑保持；两垫法向最小1.581562/1.618526N、桌面力0、平均向上承力1.961479N。arm/closure/lift执行门PASS，child return0。仿真原生时间从86.821停在87.041，已确认暂停；这不是动态稳定验收所用的时间。
用户随后明确回复“都可操作”，只记为外框与相机操作人工PASS，不推断为抓取视觉或整个长Goal审批。长Goal保持暂停。

本次输入状态不一致已实测，但最初为何漏掉松开事件尚未锁定；跨屏/切换/退出拖动仅为待证假设。不要宣称永久解决WSLg，也不要把此前COPY MODE共享内存问题与每次输入故障混成一个确定原因。


### 075407Z复现失败：将纯鼠标原生控制与键盘焦点协议分开

用户复现中 pointer preflight PASS、Windows foreground PASS，但 select-key-zero 被 XGetInputFocus=0 在发送任何鼠标事件前拒绝。容器已清理；不是IK、夹爪闭合或抬升失败。前一版只修Windows前台激活，不足以解决此路径，不能要求用户反复重跑。

Xlib手册明确InputFocus控制键盘事件（None时丢弃键盘事件）：https://raw.githubusercontent.com/mirror/libX11/master/man/XSetInputFocus.man 。现有Key初始化实际只点原生滑块，不需要任何按键。保留原默认焦点必需路径以及Backspace/Space协议；仅在显式可见固定站位审核入口采用独立 verified-pointer-only 协议。它要求PID/XID/title/viewability、Key/相邻按钮像素指纹、指针精确坐标、root-child属于目标窗口祖先、无已按鼠标或操作修饰键、最终窗口几何/真实root位置不变。无键盘事件。Pause改点原生Pause单选项，非Space切换，也验证对应radio像素。未更改机器人、安全门或直接写活跃qpos。

点击返回不等于Reset/Pause成功：仍要求native generation+1、episode初始位姿匹配、后续新鲜站立，以及Pause后真实模拟时钟冻结。缺失像素、遮挡、身份变化、已按按键、移动中的窗口全部拒绝发送鼠标按钮事件。无焦点单元测试仅在完整鼠标证据齐全时允许两个按钮事件，默认键盘路径继续拒绝无焦点。

081450Z重新执行实际RGB-D物理抓取：Key鼠标协议2个按钮事件/0键盘事件，native generation到1，STAND物理观察PASS；闭合/抬升门PASS，最小抬升57.610348mm/最后1sim秒。原生Pause鼠标协议及实际冻结检查PASS。没有再次重启图形服务，没有运行行走或矮瓶。审核窗口保留，完整进程收尾/清理在用户关闭或等待期限后单列；不能提前宣称cleanup完成。五阶段长Goal仍暂停。


### 083200Z初始化未加载与实际原生UI收件确认

用户083200Z复现中，纯鼠标校验通过、2个按钮事件已提交，但前后截图的Key均为-1，native generation始终0，根位置仍在START，最终高度约0.0825m。不是初始化成功后再跌倒，也没有进入IK、闭合或抬升。该失败容器已确认不存在。此前仅检查XTEST/XSync，未证明应用收到对应控件点击；具体WSLg丢失/重定向原因仍未锁定。失败发生在第二屏、成功记录在第一屏只是相关性，不能据此断言多屏就是根因。

现有只读native observer额外记录真实mjui_event的PID、单调时间、坐标、返回控件、section/item；原函数仍恰好调用一次，未写model/data/control，也未改厂家二进制。原生337源码支持Key变更直接pending.load_key，不应再补点Load key造成第二次Reset：
https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/simulate/simulate.cc 。

显式可见审核路径改为单次按下→最多0.75s等待真实应用Key控件press收件→finally单次松开。没有任意固定按住时长、重复点击、键盘输入或放宽身份/像素/坐标检查；超时或异常也松开。收件只证明原生UI处理，仍必须独立通过generation+1、初始位姿及新鲜站立检查。Pause同样取得真实单选控件收件，再独立确认模拟时钟冻结。默认其他Reset/键盘路径保持原协议。

离线110项测试通过，含错误PID/控件/坐标、旧记录、空/半写日志、超时释放、异常释放、禁止第二次点击；337严格编译与Python/bash/diff检查通过。证据：
reviews/issue-46/native-ui-receipt-correction-offline-20261107T091007Z/ 。

091358Z真实原生物理复核：PID30的Key控件(section2/item5)在提交后77.871ms返回，native generation到1；窗口即使XInputFocus=0也通过独立纯鼠标收件证明，0个键盘事件。站立、张开、RGB-D重新估计、手臂到位、闭合、抬升均PASS。最后51帧/1.000sim秒最低抬升58.050957mm，双垫最小法向1.640800/1.659285N，桌面力0，平均向上1.961917N。Pause控件原生收件和实际时钟冻结80.961s均通过，保留交互审核窗口。该运行只证明当前30cm/.200kg桌前固定站位的SIMULATION_ONLY源派生碰撞/控制配置，不包含从START行走或整个长Goal。

审核窗口有明确900s上限；超过时间自动清理应记录为review timeout，与已完成的抓取物理PASS分开。085313Z先前收件观测运行已完成抓取，随后900s审核到期并确认容器删除；没有假称其审核关闭属于新的抓取失败。当前091358Z审核尚未结束时不提前声称cleanup完成。长Goal保持暂停。


### 2026-11-07 Goal恢复：微调的真实资格、时钟与配置差异

用户恢复五阶段Goal，当前仍以完整行走到桌前后的受限物理抓取为最高验收；四方向微调是准备工作，不把单方向或固定站位抓取外推为整个阶段通过。README旧的“尚无50mm/1s物理抬升”已纠正；保留加强原生腿速审计后的全链路未资格结论。

095411Z在左移后零命令阶段因ROS新鲜度失败。独立连续executor本身未消除此问题；后续103145Z记录到多个源时间戳回退和桥输出约1.103s间隔。只读38s系统时钟采样实测回退0.473953s、0.613829s（micro-clock-regression-readonly-20261107T103913Z）。经用户明确同意，仅临时停止Ubuntu systemd-timesyncd；对比仍回退0.974318s、0.426214s，不能认定该服务是唯一原因。已恢复active/enabled且取消恢复定时器，未改配置或重启发行版（wsl-temporary-timesync-qualification-20261107T104331Z/restoration.json）。未确定具体调时调用者；Docker/WSL公开问题只能作为线索，不能替代本机证据。官方参考：https://ubuntu.com/wsl/docs/stable/explanation/time-sync/ 。

现有桥原样SHA保持9996c197...；只在显式micro诊断中生成SHA记录的调度适配副本，将50Hz计时器与输入过期计算改用steady/monotonic，保留ROS消息时间戳、源、数值/限幅、QoS和注册字段。不是修改MC二进制或ONNX。接收端独立观测wall-minus-monotonic跳变，只在零命令期间清除旧代次缓存并重新取得新鲜状态/完整停稳窗口；不能拼接跨代次窗口。运动中跳变立即发零并锁定失败。源内时间戳仍严格递增，旧/重复数据不刷新年龄，0.20s门不变。Humble无物理测试确认steady timer，0.5s内实际收到24个零消息，MC/仿真均未启动（micro-steady-clock-humble-smoke-20261107T105712Z）。

105949Z：原MC配置、elliptic/impratio10、0.12m/s左移1s加有界航向反馈，实际left +28.196mm、forward +38.444mm、yaw -0.026955rad；停稳1.901165s/76帧，目标最大范围0.001115rad，PASS。零命令观察经历2个时钟代次，最终窗口在同一新代次。它是耦合方向能力，绝非纯横移/精确3cm定位。
110238Z：相同条件右移，DDS观察到47个-0.12m/s横向消息，源/LD已观察，但最终left +21.059mm、forward +26.846mm。方向FAIL、停稳PASS（1.988014s/73帧）。没有把发布或LD读回当作正确运动。

114131Z：原MC配置的0.12前进在约0.4457s出现实测pelvis速度0.259115m/s，按0.20门中止，无自动重复。后续核对发现当前native入口未挂载旧Issue80诊断覆盖层。现增加显式 --reviewed-foundationwalk-profile，仅复用已批准的冻结文件6eb71287...：左右膝动作缩放0.25->0.19、min_run_velx 0.5->0.1，其他YAML值不变；后者厂商语义仍未知，不据字段名断言原因。源配置b3081e9b...、ONNX b8cca5e9...、runner 0fa34f81...、robot_model 614b0644...均验证，实际容器挂载SHA另外核验。原始文件不改，不生成新调参值；始终标记SIMULATION_ONLY/MODIFIED_FOUNDATIONWALK_CONFIG，历史通过不等于本体当前资格。

115241Z复用旧覆盖层后1s前进净forward -11.189mm，停稳PASS，方向FAIL。115948Z单独标记目标3cm/<=2s诊断，0.12输入仍未到目标，净forward -5.994mm，最终误差38.768mm；停稳PASS、目标FAIL。新2s入口在原M1<=5s内独立限定，默认旧micro仍最多1s，速度/0.18m工作区/50mm横向/姿态/关节/物理停止门不变，目标入口额外要求20mm最终误差与目标触发停止。

Run81报告文字称转向降到0.12，但原始velocity_command_trace有223条非零命令全部forward=0.14，包含angular=0.075，0条0.12。刚才按文字加入的降速分支导致120826Z实际76次0.12、仅9次0.14，已经依原始数据纠正。121732Z保留0.14仍未在2s内到3cm，不能归因于这个分支一种原因。

只读下肢模型对照（native-mc-leg-frame-equivalence-readonly-20261107T121142Z）：源/当前下肢关节轴、位置、限位、身体惯量无差异，pelvis差异为已知场景平移。机器人子树质量分别43.474797kg/44.308583kg；376.99kg是整个场景总质量，含静态环境，不能当机器人质量。数值选项差异为cone 0->1、impratio 1->10；timestep=0.001、积分器、solver/iterations/tolerance/gravity相同。该抓取数值覆盖是全局选项，也作用于脚底，尚不能证明它就是行走问题原因；正在单独恢复native pyramidal/1做不同假设对比。

122629Z该对比在MC启动前因native首帧陈旧中止；实际GL为RTX4060，原生启动记录最大间隔1.355s。现只在MC启动前增加最多4s等待两帧同PID/同代次且各自<=0.20s、模拟时间推进的就绪检查；不放宽运行期新鲜度。异常身份/缺body不重试；回归覆盖陈旧、冻结、PID/代次变化。

小步入口亦接入原生Reset控件收件确认，严格PID/XID/title/pixels/pointer/ancestry；只发一次，不因未确认补第二次。111252Z在发任何Reset前因X11按键滞留退出。确认无实验/其他X11 GUI后，仅恢复WSLGd监管的Weston（12110->39907），发行版和Docker未重启，指针前检实测PASS（wslg-left-button-stale-recovery-20261107T112037Z）。临时/tmp的旧Xvfb工具已不存在；865984字节Ubuntu包仅按APT元数据SHA解包到持久reviews，未系统安装。隔离GPU显示尚未验证成功，不宣称可以替代WSLg。

以上所有已终止native容器均有独立absence证明。四方向精确微调、150mm动态抓取、感知站位闭环及LiDAR替代仍未资格；当前Goal继续，不标完成。


123648Z恢复pyramidal/1后仍未到目标，在启动约4.353s（已发零并请求STAND）发生yaw=-0.120073rad而中止，不能把全局求解设置单独认定为根因。原生非地面受力接触未见机器人自碰撞，只有瓶-桌支撑。源/当前IMU站点imu_0及安装变换也一致。旧Run81前2s实际forward约+93mm，而121732Z在同样0.14输入的前2s约-13mm；0.5..1.9s新MC髋俯仰目标范围约0.02..0.03rad，旧记录约0.14..0.19rad，不能当缓存未更新或简单符号翻转。

下一项只读诊断复用旧Issue80 McRLDebug记录器及其已构建ROS覆盖层，不执行旧运动脚本。MC消息introspection库仍为935d4226...，字段为header/obs_vec/action_vec；只提取原快照函数，原记录器源码不改。真实Humble无物理构造确认仅一个RL debug订阅、0个action client、0个速度publisher（只有标准parameter_events），实际schema匹配。证据 reviewed-mc-rl-debug-humble-smoke-20261107T130324Z。显式 --capture-mc-rl-debug 要求当前唯一真实MC publisher身份与实际类型包到达后才允许该诊断运动；向量索引只依匹配配置解释。Micro另补原始IMU、ROS odom及leg command header，零/非单位IMU四元数拒绝，不能以新鲜但零填充的消息通过。


132410Z修复debug图发现等待后，实际唯一MC发布者mc_ros2_node158已确认，采集到SD的85/14和LD的111/29向量。匹配配置的command槽0..2在0.14/0/0.075外部输入时观测为约0.126/0/0.1125，零命令后为0；不据此猜测内部缩放字段语义。它支持“非零/零已进入策略命令槽”，仍不是显式MC收件ACK或物理运动/停止证明。IMU、ROS odom与native yaw同时对照一致，四元数有效，不能盲目翻转坐标符号。

该数据还显示显著的全身执行差异：当前外部保持器把肘部维持约-1.14rad，旧成功行走约-0.36rad；旧肩yaw约±0.6..0.7而当前接近0。135741Z仅恢复到私有观察话题的MC arm publisher（没有改变实际arm控制），得到557帧SDK输出，其中447帧正增益：SD为0/0，LD后出现原厂50/3等目标。旧“在SD先等MC双臂正增益再允许非零”的原型前提不成立，不能把这个失败认作缺少MC能力或厂家机械资料。

原生arm中继随后按真实SDK状态修正：以原厂策略default_dof_pos的14轴参考姿态及已有bootstrap增益保持SD上肢；LD有效正增益输出原样转发；回到inactive时以已有源控制保持测得姿态。没有改MC/ONNX、机器人、增益/力上限，也没有qpos写入。源参考姿态21点静态过渡无新增自穿，COM到支撑凸包最小57.85mm，最小源关节位置余量0.261rad（source-mc-reference-arm-posture-static-20261107T135951Z）；仅静态资格，非动态成功。初次纯inactive透明转发暴露腕部被动下垂与源限位拒绝，不能删除物理限位来通过。

新增模式继续检查实际全29轴源速度/位置、真Root速度与姿态，并检查活跃受力自碰撞/非足地面接触。零SD增益只计source transport，不计主动控制；LD时必须同时观测到实际原厂正增益源。无物理Humble已执行实际SDK初始化/张开消息构造、保持参考姿态构造和退出验证，真实机器人发布次数0（source-reference-idle-native-ld-humble-smoke-20261107T150902Z）。当前正在进行一次不同控制交接条件的有界物理验证，尚不宣布四方向或整条Goal通过。

### 原生上肢接管实测与观测覆盖修正（2026-11-07 15:57 UTC）

151039Z在运动前因旧消费者必需的last_unique_source_receipt_s字段遗漏退出；修复后以真实Humble producer/consumer夹具核对，未放宽0.20s新鲜度。152107Z实际LD正增益上肢接管成功，但0.14输入在首个非零后0.424857s出现native根平移速度0.251271m/s，按原0.20门中止。153630Z单独对比既有最低0.12输入，仍在0.315094s出现0.228874m/s，13次非零后即零速并终止；未达3cm、不允许rearm，两个容器均确认删除。不能声称上肢接管已经解决行走或完整Goal。

发现旧C observer的contacts数组只输出瓶/桌相关接触，虽然active_contact_count是全模型总数。之前“没有其他机器人受力接触”的表述因此不足以证明全模型无自碰撞；新micro接触guard此前缺少完整捕获的前提。本次修正只读采集：输出全部已编译接触、显式ALL_COMPILED_GEOM_CONTACTS及全部actuator ctrl/force和每轴qfrc_actuator/constraint/passive/applied。微调入口必须捕获scope正确且记录条数等于ncon，否则运动拒绝。未改变碰撞过滤或物理状态。

用实际3.3.7缓存绑定对153630Z五个保存的qpos做零步静态重建，完整接触均为单一raicom_floor与脚、瓶与桌，无重叠地板或机器人自接触；首次速度超限时右脚最大几何进入约4.313mm。它仅是静态几何证据，缺少当时ctrl/freejoint速度，不能从重算力推断真实冲量因果。证据native337-abort-all-contact-static-audit-20261107T154453Z。下一项应比对真实MC腿目标与实际native力矩/接触变化，不能继续盲调速度。

新只读C observer严格编译通过；3.3.7独立小型合成夹具实测两motor ctrl=.25/.10，qfrc_actuator=.35且全接触记录完整（native337-force-contact-observer-window-fixture-20261107T155710Z），未启动X2/MC。最初单步夹具取到了mj_loadXML内部模型校验产生的首帧，修正为检查同一受控数据后续0.021s帧；失败和解释均保留，非ABI不匹配。138项聚焦测试通过。时钟服务重新确认active/enabled；没有继续停止或更改配置。

### 取送到凳区新Goal（2026-11-08）

桌与椅坐标可用已知地图真值；先完成RGB-D取瓶、持物到椅区并停稳。椅座面实测网格高度约0.328215m，整个椅单mesh凸包在中心表面0.536048m，差208mm；未改变场景，不把空气支撑计放置成功。fetch-delivery-chair-geometry-20261107T173112Z保存了准确几何。

161319Z的Reset确已送达，事件后68.342ms出现sim=.001/gen2；dispatch前最近真实样本为gen1。新确认根据同PID、337、dispatch前0.20s内的最近原生样本、真实Reset控件press及恰好两个按钮事件，要求事件后恰好+1代次、<=.05s的重置边界，并独立要求当前新鲜同代次状态。不会补点、放宽身份/新鲜度或写位姿。完整保存trace重放PASS；原键盘与固定站位已通过入口不变。最初tail8MB重放遗漏更早事件，改为流式读取dispatch附近窗口，保留先前失败；该读取不改变控制时钟。

### 抬升后的真实收臂准备

增加显式 --post-lift-retract-world-m 诊断，沿用现有包持圆IK、tracker和原力反馈闭合。默认抓取脚本不启用此阶段。新阶段必须先通过原50mm/1sim秒无支撑承力门，再以4s三次平滑曲线执行<=250mm平移，随后1.5s保持并再次评估相同物理门；对象相对手的滑移、姿态、接触、29轴源位置/速度及root0.20m/s限制均保留。仅该显式阶段增加6s有界child预算，不改旧入口。无object pose活跃写入或夹持约束。

现有真实抬升末端的-150mm世界X/-150mm世界Y路径，13点静态IK和瓶/桌碰撞筛查通过（fetch-chair-postlift-retract-static-20261107T175409Z）；实际CLI前后调用链亦通过APPROACH/LIFT/POST_LIFT_RETRACT_SWEEP（fetch-chair-retract-cli-preflight-20261107T181259Z）。这些不是动态收臂或持物行走PASS。

174412Z新Reset收件与边界确认通过，但bridge图尚未看到publisher时启动检查超时，非零命令0；容器确认删除。父流程原先在ADD响应打印时即把bridge视为启动，现在等待其真实started日志，仍由子流程独立证明唯一图/QoS/新鲜实际输出和源/模式。ADD任务完成仍未确认，不能把state=0当完成。

181830Z真实RGB-D抓取仍完成50mm/1sim秒无支撑保持（最低57.696mm，双垫最小1.671/1.661N，桌力0），收臂刚开始约4.6mm指令时出现CLOSURE_PHASE_EXPIRED，非掉瓶/IK失败。原因是RETRACTING继承了LIFTING起始时间，却落入6s阶段期限。修正为独立且仍为6s的RETRACTING阶段，只能在新鲜物理抬升门通过后的0.20s内建立；每帧更新目标不刷新该起点。未延长旧阶段或状态新鲜度。另将“抬升已完成”和“后续收臂失败”独立分类，保持总体失败；完整行JSON仍严格校验，只忽略正在写入的末尾半行。实际容器已清理，未进入行走。

### 持物到椅区交接实现与预检

182833Z实际收臂PASS：位移(-148.435,-148.774,-2.137)mm；随后51帧/1sim秒最低抬升54.181mm、双垫最小1.613/1.588N、桌力0。收臂阶段所有源29轴速度未超限，最大比率0.04350；native容器已确认删除。此后才实现CARRY_HOLD，不会把该运行说成持物行走。

新显式carry-goal-world-xy入口复用同一HAL手臂/夹爪所有者，固定最后实际发布的承力关节目标并继续力反馈，不读瓶真值来追逐移动目标、不写qpos。仅新搬运阶段不再用“全程base漂移<=50mm”误判合法导航；原站立段仍单独按相同阈值验收，全部轨迹保留。世界导航复用现有PointGoalController以及Micro的身份/QoS、独立反馈executor、零命令、STAND与原复合stop gate。只有已夹持的两个末端pad/test_cylinder对允许接触，其他碰撞和源29轴限制不变。

新SIMULATION_ONLY_FETCH_CARRY_ROUTE显式上限90s任务运动、forward0.12、angular0.075、root0.20、状态年龄0.20、到椅区基座误差0.15m。该全房间点目标不是旧0.18m micro或已资格的50mm参考线跟踪，原微调/曲线分支不改；新路线记录实际对直线偏差并持续核对房间边界、真实碰撞、模式/源与握持。无实测位移/角度改善3s即停止，不盲等90s。仅显式source-clock epoch异常允许先零速、请求并确认STAND、完整原复合停稳通过后，以新鲜反馈重新规划；最多4次且不重置总deadline。真实物理/保护/遥测异常不重试。默认旧micro仍遇活动时钟跳变即失败。

真实Humble无物理smoke已完成两个实际ROS节点构造/销毁，握持owner为0client/0velocity pub；导航为既有4client、1个cmd_vel publisher、实际发布0次，executor退出确认。没有启动MC/physics，证据fetch-chair-carry-humble-no-physics-smoke-20261107T191709Z。模拟控制流回归验证目标触发zero-before-STAND/stop、物理fault不重武装、时钟异常必须停稳、deadline不被重武装。完整真实取送尚未运行。

194406Z再运行：相机抓取/收臂又通过，抬升后保持最低55.131mm，但导航尚未非零就因CARRY_PUBLISHER_NOT_UNIQUE拒绝。进一步分开时间线：导航拒绝约ready+2.35s，握持owner约ready+6s才因对象相对手偏移10mm退出；二者独立，不能把后者当图失败的起因。收臂后1s成功尚不足以证明持续搬运。此episode容器确认删除，非零命令0。

精确配置MC的无MuJoCo/无action/无速度发布检查证实：enable:false的MC arm/hand publisher仍出现在ROS图，与HAL owner各形成2个公开endpoint（fetch-chair-mc-disabled-publisher-graph-20261107T200452Z）；不是任意前缀/节点命名问题。新carry配置同时隔离两个PublisherManager topic与对应AimRT pub_topics_options到私有inactive命名空间，enable仍false；腿/腰和厂商原配置不变。坚持公开话题严格唯一，不放宽检查。固定站位默认配置不变。

另修正carry代码生成插入位置，恢复HAND_CLOSE_HOLD发布次数统计在原hand发布分支；此前计数被错误缩进到CARRY分支，造成false“未闭合”报告，实际物理握持证据不受影响。补结构回归。后续先要求8个实际simulation秒、当前native/握持heartbeat持续新鲜的静态保载（12s wall cap），再允许非零。MuJoCo3.3.7官方XML文档说明noslip后处理用于抑制软接触摩擦维度漂移；复用已存在、单独标记的noslip_iterations=2诊断，不改摩擦、质量、几何、约束、控制增益或力上限。只是待验证假设，不是对真机摩擦/抓取的证明：
https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option

完成隔离后，等待真实MotionControlModule Started Succeeded再进行6s无物理ROS图检查，公开arm/hand各恰好1个HAL owner，MC只在两个私有inactive话题各注册1个endpoint，速度publisher为0；无action/无physics，audit归档后容器确认删除。证据fetch-chair-inactive-channels-ready-graph-20261107T201903Z。因此不会通过“接受多个公开publisher”来绕过控制权问题。

202941Z NoSlip2物理运行完成RGB-D抓取、抬升、收臂并写出carry_ready，但8s保载观察函数把require_fresh_sample返回的验证metadata当成含source stamp的记录，因host_monotonic_ns字段缺失在任何行走前中止。不是掉瓶或MC错误，数值诊断的长持握效果仍未完成。给验证metadata增加真实原始host_monotonic_ns（不改年龄阈值或状态），并让完整父交接分支测试直接读取真实JSON/真实freshness函数，避免mock掩盖API形状错误。原失败与容器删除证据保留。

203849Z在动作前POST_LIFT_RETRACT_SWEEP拒绝，最后点位置0.254mm、加权轴残差0.001312，无碰撞；没有闭爪或行走。仅将离线收臂求解预算90->200并记录nfev/status/optimality，门槛和源关节界不变。同一保存状态复核找到通过解（endpoint位置2.79e-11m/轴残差9.99e-11；37次endpoint评估），见fetch-chair-retract-solver-saved-state-check-20261107T204558Z。该结果证明可行解存在，不把它等同动态执行或放宽残差。

205142Z实际通过8.0模拟秒CARRY静态保载及公开唯一publisher/QoS图；移动前原复合停稳PASS（实际采样1.916940s/85帧，nominal2s、保留原min1.90s）。随后PointGoalController在首个运动循环返回stale_state，非零发布0。原因可定位到now=tick在pump/sample之前取值，而原生样本可能在这期间推进，造成合法新样本“received time > now”。改为在获取样本后读取真实单调时间传给控制器，不重打源时间戳、不放宽0.20s；加入pump延迟后仍应合格的实际控制流程回归。NoSlip2的8秒保载通过单列，仍不是搬运/真机PASS。容器已删除。


### 2026-11-07 携带转向的独立失败与最小修正

- 205948Z：八秒真实静态夹持通过；开始角速度 0.075 rad/s 后
  +0.348 s 瓶轴倾斜首先超过 0.10 rad，此时相对虎口滑移 2.94 mm、
  根部速度 0.121 m/s。旧携带分支冻结关节；退出后失去夹爪发布者。
- 213524Z：携带已改为现有 IK 的机身航向坐标位置、世界竖直轴任务。
  仍在 +0.33 s 因 pelvis/left_hip_roll_link 接触穿入超过 1 mm
  停止，记录法向力约 604 N；瓶子倾斜是后续独立失败。
  不以静态 IK 或先前空手行走证明持瓶转向成功。
- 修正包括携带目标的源时间差分关节速度前馈；原分支一律发送
  velocity=0 不适合跟踪随机身摆动变化的目标。前馈限定在源关节速度内。
  异常先撤销携带健康标记，让底盘置零，再最多五秒重复最后合法夹持命令；
  不计算新异常恢复姿态，不把最后命令发布视为物理停止。
- 原 .075 rad/s 仍是默认诊断。另一个显式
  --carry-sdk-minimum-turn 仅采用 pinned raicom2026-agibot-X2
  mc_locomotion_velocity.py 给出的最小非零角速度 0.10 rad/s；
  它是不同的仿真输入配置，根部 .20 m/s、所有源关节/接触/姿态门不变。
  该例及参数中的 cmd_threshold 不能单独证明控制器内部行为，
  故后续使用版本匹配的只读 McRLDebug 观测，不继续猜阈值。
- NoSlip2 是现有显式数值诊断，作用于整个接触求解；
  不能把其静态抗漂移效果视作行走已兼容或真实摩擦认证。
  MuJoCo 3.3.7 的 option/noslip_iterations 文档将其定义为主求解后的
  摩擦方向后处理：https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option


### 原生 LS 的零指令准备态（231511Z 实测）

LOCOMOTION_STEP 请求完整返回 code=0/status=1/message=success；
CommonState.speed_status.speed_mode=4，MC 日志明确记录进入 STEP 档。
在零速度下 GetMcAction 与 CommonState 仍一致为 STAND_DEFAULT/100。
因此 LS 的“已准备”与“已开始步行”必须分开：允许上述正向证据、
原身体/关节/新鲜度/停止合同均成立后才发有界输入，随后必须确认
实际 LS 与选中源；若实际返回 LD、保护或不一致，仍停。231511Z
没有发送任何非零速度，不能当作 LS 行走失败或成功。

此替代模式的 cpgstep_v1.onnx SHA256:
67f9e7e3f40cbf7809c291a224a7183c5d553d81931ba9509ff094ae237d2612。
策略与配套 YAML 未修改；它是不同于 FoundationWalk 的独立方案，
不替代、重标或删除旧 FoundationWalk 失败。其完整 17 个动作关节
均存在于冻结场景，源动作表允许 SD→LS→SD。


### 接触数值选项与携带力余量的隔离

NoSlip2 的作用域是整个模型，包含脚底；旧静态夹持 PASS 不能证明
同一求解选项下的步态已兼容。独立的下一配置恢复 noslip_iterations=0，
保持模型/接触摩擦、质量、MC/ONNX/增益、物理速度与源关节限制，
仅在完成原有抬升与收臂证明后，将携带法向力目标的重量余量系数
由 1.5 改为 3（H300/.200kg、mu=1 时约 2.943 N/侧），根夹爪执行器
仍受原 2 Nm 限制。这是显式 carry-grip-force-margin=3 参数，
默认抓取/抬升与其他命令仍为 1.5。必须先通过八秒真实静态持瓶，
不能把力目标、Python 发布成功或静态 IK 当成受力和行走验证。


### 2026-11-08 精确待审边界与下一步

完整取送仍未通过，不把固定桌前抓取、回放或静态IK写成START→凳子成功。

| 独立现象 | 实测/状态 | 证据（reviews/issue-46/） |
|---|---|---|
| 原生LS实际启动 | 28次角速度0.1发布；Get/Common均LS、选中合法源，McRLDebug观测长度65；+.558s根速度0.2234m/s越过原0.20门 | raicom-stl-xvfb-native-reset-stand-20261107T232631Z |
| 关NoSlip、携带力目标3 | 实际每侧约3N，轴向下滑拟合−0.4359mm/s，10mm滑移门失败，导航NOT_RUN | raicom-stl-xvfb-native-reset-stand-20261107T234059Z |
| 待审局部数值方案 | 仅22对已启用包胶/瓶子接触 solreffriction:0.02→0.002s；正常参数、mu、质量、全部执行器/关节/停止门不变；该历史提案随后获C.2批准并已实测，见下文 | fetch-chair-local-friction-diagnostic-proposal-20261107T235658Z/proposal.json |
| 椅子源组件 | 14个连通实体，seat组件6的顶面0.328213m；各部件凸包在座面中心仅seat提供0.328213m上界；整椅凸包原0.536048m | fetch-chair-source-component-seat-audit-20261107T233753Z/result.json |
| 放置可达性 | 3个固定站位手臂解、3组保守腰姿态及全源边界离线解未找到合格位置/轴；无全局不可达结论 | fetch-chair-seat-fixed-stance-ik-20261108T000057Z/ 与 fetch-chair-seat-waist-reach-diagnostic-20261108T000301Z/ |

上述是235658Z时的历史提案状态；随后直接批准已记录在
model_integrity.md C.2，实际加载及持瓶结果见下文。原模型和失败记录保留。
当前未批准的是独立运动态速度门提案，不能由接触参数批准推导。

整条任务仍是Python研究原型加原生C++MC；未收敛正式C++节点，未完成持瓶
返回凳子、放置、全链路LiDAR替换或真机验证。实验代码继续保留在
scripts/research/，不提升到正式运行模型。PR保持Draft。

#### 已获准的完整接触参数差异（物理验证中）

源场景 SHA256: a310757ec1d8d82f604b59b08c3e0f98f6cac43e71ade99a3acd52298c0d07fc

原输入是已启用的动态配对，摩擦维度继承normal solref=0.02s；下列片段已按2026-11-08直接批准写入独立验证模型。其每对 normal solref/solimp、mu、condim、margin/gap均逐项核对一致。与全局NoSlip不同，它只针对现有两侧包胶代理与瓶子。直接批准已记录在 model_integrity.md C.2。离线验证392项非配对数值数组及全部既有命名ID保持相同，真实物理结果仍须另行验证。

```xml
<contact>
  <pair geom1="R_hand_narrow3_Link__collision_coacd_0" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_0" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_1" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_1" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_2" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_2" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_3" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_3" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_4" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_4" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_5" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_5" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_6" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_6" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_7" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_7" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_8" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_8" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_9" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_9" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_10" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_10" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_narrow3_Link__collision_coacd_11" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_11" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_0" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_12" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_1" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_13" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_2" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_14" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_3" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_15" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_4" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_16" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_5" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_17" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_6" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_18" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_7" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_19" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_8" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_20" solreffriction="0.002 1.0" />
  <pair geom1="R_hand_wide3_Link__collision_coacd_9" geom2="test_cylinder_collision" condim="3" friction="1 1 0.0050000000000000001 0.0001 0.0001" solref="0.02 1" solimp="0.90000000000000002 0.94999999999999996 0.001 0.5 2" margin="0.0" gap="0.0" name="diagnostic_pad_bottle_21" solreffriction="0.002 1.0" />
</contact>
```


### 已批准局部摩擦对比的首次真实加载（004533Z）

原生337模型正向确认22对、tau=0.002s、NoSlip0；闭合双侧接触通过，
但8mm载荷转移探针中出现单侧接触丢失，未进入携带。
提起前每侧仅约1.6N：原 carry-margin=3 直到抬升/收臂后才使用，
没有在实际承重前建立该预紧力。控制修正将已显式选择的3倍重量余量
用于本局部参数配置的闭合/抬升，并在探针之前要求真实连续测得该力；
默认配置仍1.5，现有2Nm、关节/速度/姿态门和0.002s参数均不变。
这项控制修正不把首次局部参数试验改写成成功。


### 夹持目标与承重门的区分（010516Z）

提高闭合目标后，末段25帧实际法向力范围为窄侧2.119–2.437N、
宽侧1.930–2.280N；因新加的“必须等于2.943N目标”门而未执行抬升。
这不是零接触。控制目标3倍重量余量与既有1.5倍重量承重预算必须分开：
minimum_load_normal=1.4715N/侧，target_normal=2.943N/侧。
闭合继续追踪较高目标，抬升前验证真实连续最低承重力；原0.1N失联、
10mm滑移、50mm真实抬升和所有关节/姿态/停止条件均不变。
旧010516Z的未执行状态保留，后续必须用真实物理验证，不能回填PASS。


### 011616Z：有效物理持瓶窗口与外部运行环境中断分开

最后有效原生轨迹的8.000s/401帧内：最低离台61.263mm，窄/宽法向力
最低3.096/3.120N，桌支撑0，根速度最高0.00002095m/s；
实际NoSlip0、22对tau0.002s。无完整导航结果，不回填路线PASS。
随后WSL/Docker运行环境单调时钟从约133994s降到几百秒，容器Exited255，
OOMKilled=false，部分新结果文件为0字节或含NUL；重启发起原因未知。
原文件保留，合法前缀5251条、首次损坏行5252及原SHA另存
reviews/issue-46/fetch-chair-wsl-restart-recovery-20261108T013357Z/。
该容器已按精确evidence bind核对、删除并确认不存在，其他历史容器未动。
Git fsck通过。一次性ready/result改为原子写入、fsync文件和目录；
50Hz控制/心跳不增加同步I/O。独立运动审计改为流式逐帧读取，
保留全部边界/代际/回退校验，避免把整个原生轨迹同时放进内存。
这不证明重启由内存导致；仅降低审计资源占用。


### 严格运动态门的独立审查（015650Z离线取证，未变更）

完整源轨迹分开检查指令段与置零制动段：
- LS/013934Z：指令0.5267s，0.5499s开始零指令；指令段根速度峰值
  0.2058215m/s、制动峰值0.4323874m/s。最低根高0.650956m、身体倾角
  最高0.069704rad，全部源轴最高速度比0.51694，没有源位置或禁止碰撞
  越界；指令段瓶轴世界倾角0.02943rad、手/瓶轴相对角0.01671rad。
- FW/014728Z：指令0.3385s，先出现瓶轴倾斜0.12363rad；后续制动还
  记录到左肩roll位置越界与左手/髋接触，不能因root速度门审查就重标PASS。
- 两次原结果均保留FAIL。0.20m/s瞬时pelvis门来自
  native_base_micro_probe.py研究诊断，不是URDF自由根的厂家速度限位；
  SDK输入速度范围也不能直接当实际身体速度的验收证明。

已提出独立SIMULATION_ONLY_TRANSIT_050策略，但未实施：
仅有界导航/制动阶段root瞬时门0.20→0.50m/s；
命令≤0.12m/s、角≤0.10rad/s、站立/停止/源关节/姿态/夹持/
碰撞/时限门均保留。需具体批准；不能凭完成Goal自动放宽。
完整数据与提案：
reviews/issue-46/fetch-chair-transit-guard-forensics-20261108T015650Z/
reviews/issue-46/fetch-chair-transit-speed-policy-proposal-20261108T020015Z/


### 未通过的携带夹持平面轨迹已退出运行入口

在等待运动态策略审查期间，仅离线检查了更紧凑的携带姿态。
单端点可求解，但完整分段轨迹在原位置/轴/旋转精度合同下失败：
023824Z联合路径和024526Z先平移后旋转路径都未通过。
未放宽残差，没有对此运行mj_step或持瓶运动。

四个仅属于本次原型的源文件差异已逐项保存SHA256和完整patch后，
恢复到20e43c9c8ef07b4c3961fcbb5928284105afdf4b。
原已通过的抓取/抬升/收臂、本次C.2参数及其他用户文件均保留。
该原型不作为可执行携带方案：
reviews/issue-46/fetch-chair-unqualified-plane-prototype-archive-20261108T024923Z/disposition.json。
恢复后的原范围182项离线回归全部通过；这不增加任何持瓶导航成功声明。

### 异常退出的停止证据显式分层（离线修正）

013934Z与014728Z在携带健康门失败后立即置零，执行既有1.05s
零发布清理；两轮都没有STAND_REQUEST，也没有post_stop_composite。
因此停止合同是NOT_RUN，不能把它写成物理停止PASS，亦不能由已发零命令
推导MC收到或施加零命令。首个物理失败仍保留，未经新实验改写。

现有携带runner只增加结果级stop_evidence：末次非零/首次零发布时间、
零发布跨度和间隔、STAND请求及原响应、新鲜且严格推进的SD读回数量、
既有独立停止门的PASS/FAIL/NOT_RUN。该报告不产生重武装许可。
摘要自身异常不覆盖首因，也不阻止原置零清理；运行顺序、时限和安全门未变。
六项聚焦回归加入原范围，合计188项通过；没有启动新的物理实验。
离线结果及原文件哈希：reviews/issue-46/fetch-chair-abort-stop-evidence-20261108T031451Z/saved_trace_analysis.json。

### C.3已批准与首次运动态实际结果（044753Z）

维护者已直接批准独立TRANSIT050仿真策略，范围记入model_integrity.md C.3。
默认及无有效许可时仍0.20m/s；只有导航PID/启动标识、Reset代际与单次
有界许可全部匹配，才在导航/制动时使用0.50m/s。原源轴/姿态/夹持/
碰撞/独立复合停止门不变，旧0.20审计另存，不重标旧结果。
043437Z与043940Z均在参数处理阶段退出、没有创建仿真容器；外层转发
与原生层overlay字段核对已修复并加入真实解析路径回归。

044753Z实际抓取、抬升、收臂及8.000模拟秒持瓶通过。非零仅角速度0.10，
没有完成到凳子。Get/Common与合法源曾同时LS，随后13540.947至13541.123
的一次同步graph检查阻塞约0.175s，未泵取新Get响应；下一轮握手新鲜度
失败后置零。独立原生轨迹又在92.341模拟秒发现左肩roll=-0.062655rad，
低于原-0.061rad界限加既有容差，因此本轮FAIL保留，停止门NOT_RUN。

最小软件修正：保持原每秒图核验，但放到独立只读线程，控制循环继续
收样/发布；图检查0.20s超时或身份异常仍立即置零，不放宽读回新鲜度。
另发现携带阶段只给右臂原生bias力，左肩实际静态-0.04017rad与
目标-0.02738rad差约0.0128rad，kp100对应1.28Nm，恰与原生bias约1.28Nm一致。
仅CARRY_HOLD让左臂复用相同新鲜原生bias前馈；目标、增益、限位不变，
非有限bias仍拒绝，下一次真实静态持瓶门必须重新通过。
这两项修正尚未被后续动态结果验证，不能据此宣称恢复。

### 050033Z：真实夹持姿态门与紧凑收臂预检

异步图检查实测耗时0.03365s，未再因其阻塞导致握手失败；
Get/Common与合法源同时LS。约1.704s后瓶轴0.104rad触发原0.10rad门，
当时滑移1.503mm。指令段原生root峰0.47910m/s，在本次获准0.50内；
左肩roll范围[-0.05927,-0.00460]rad，当段未越源位置界限。
这不能证明左臂bias前馈已达到期望补偿效果，尚未据此改变增益。
退出后的停止门仍NOT_RUN，独立原生审计在许可撤销后发现PLANAR_SPEED；
两者均保留FAIL，不能写成安全到点。

原始McRLDebug采集在MC启动时开始，90s上限早于本轮103.9模拟秒的运动，
没有覆盖携带。保持该采集器原90s上限，改为通过八秒真实静态持瓶后、
首次导航前启动同一只读采集器；微移诊断的原启动点不变。

下一不同假设只使用既有收臂参数[-0.24,0,0]m，减小持瓶前伸/横向力臂，
不增加求解自由度或改姿态门。完整离线保存时钟CLI在
approved-circle-standing-ik-20261108T051153Z通过抓取/接近/抬升/收臂扫掠，
19.231s，零物理步。此结果不是搬运动态通过；新鲜实际状态仍须重新规划。

### 051833Z：摩擦锥饱和与下一项力控对比

紧凑收臂的真实持瓶与八秒门通过，但约1.298s后原瓶轴0.10rad门失败。
最后有效接触从静止4点变为2点，法向1.6547/1.8658N，两点摩擦椭圆
利用率均为1.000000；瓶轴倾角0.11518rad、手轴世界倾角0.05699rad，
手/瓶相对角0.08542rad。匹配本版本CPG配置obs_index.command=6与
run_scale_yaw=1.6，实际65维debug中command[6:10]=[0,0,0.16,0]，
置零后回到0；不是仅凭Python发布推断MC收到。

下一项只将显式carry-grip-force-margin由3改为6，对应目标5.886N/侧。
原最低承重1.4715N/侧、2Nm上限、材料摩擦、接触求解参数、网格/机构、
0.10rad瓶轴门及其他原门不变；默认目标仍1.5倍重量余量。
静止约3.1N/侧时原生根力矩约0.39024Nm，仍有2Nm内的执行器余量；
此估计不证明动态力矩可达。必须重新实测预紧力、抬升和八秒静态持瓶，
不得把较高控制目标或默认静态budget当动态抓稳。
证据：reviews/issue-46/fetch-chair-contact-force-saturation-20261108T052701Z/analysis.json。

### 052907Z：力控对比后的首因改变

margin6的新鲜抓取、抬升、紧凑收臂与八秒持瓶通过。实际启动原生LS后，
约2.860s的首因是right_hip_yaw_link与原桌子碰撞；随后瓶轴门也失败，
不把更晚夹持异常覆盖为首因。该局部模式纯转产生明显平移，未到凳子，
不得声称支持原地转身或完整导航。所有碰撞/姿态阈值保留。

C.3批准的问题本身没有限定MC模式，只限定导航/有界制动与保留其余门；
据此将同一显式许可用于已有的两种固定模式（原生STEP或既有FoundationWalk），
而非把实现时的额外STEP限定当成新的权限要求。下一项切回既有
FoundationWalk转向路径，保持紧凑收臂、margin6、C.2与C.3及同一源/模式/
物理停止合同。模型、二进制、ONNX与冻结诊断配置不改，不作速度扫参。

### 原地纯转不能作为取送前提；复用现有控制器的后退弧线方案

054451Z已通过新鲜启动门，实际FW源/模式生效后约0.435s触发
CARRY_HEADING_RESPONSE_DIVERGED；不盲目反转Twist符号。官方AimDK
locomotion文档（latest1.1.0与0.9.0接口字段）均定义逆时针为正，
sim_contents明确不保证仅角速度启动原地转向。latest文档不被当作已证明
匹配所装May22二进制；它与本地两模式的实际结果共同说明不能把纯转
当成前置已验证原语：
https://x2-aimdk.agibot.com/en/latest/sim_contents.html
https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/locomotion.html

独立--carry-reverse-moving-arc仅复用原PointGoalController：
用yaw+pi表示后退行进朝向，实际Twist.forward=-0.12，角速度仍原符号、
上限0.10；转向时也保持有界后退，不先做原地纯转。
0.10m位置触发置零，比最终0.15m区域验收更紧；凳区任务不要求面朝凳子，
故此模式不声称末端朝向/放置对接资格。默认原有姿态目标模式不变。
完整原STAND及复合物理/控制目标停止门、方向背离/无进展/源/碰撞门均保留。

从实际052907Z持瓶姿态作名义运动学与整个已加载场景的静态接触扫掠，
27.5s到(-0.45908,0.90071)，目标(-0.45,1.0)距离0.09971m，
无禁止穿入；没有mj_step，不是动态PASS：
reviews/issue-46/fetch-chair-reverse-moving-arc-static-20261108T055443Z/result.json。
运行中仍以真实native位姿、真实命令路径与原生全接触/源关节保护验收。

### 060414Z与非抓取臂源范围内停放

后退弧线已发实际负向线速度与转向，约0.507s因原左肩roll下限停止。
不放宽-0.061rad源界限。既有左目标约-0.027rad，静态实测约-0.040rad，
加步态扰动后越界；先前加入effort字段不足以证明原生驱动实际使用该前馈。

只在已有四秒收臂段，同时将非抓取左肩roll目标规划到源下限+0.15rad
（本源为+0.089rad），其他左关节目标不变。单轴变化<=0.20rad，三次平滑
插值；运行起点必须与预检相差<=0.03rad，当前源范围在执行前再次逐项核对。
增益、限位、执行器和模型均未修改。

新的完整保存时钟CLI通过抓取/接近/抬升/收臂及新增左侧自碰撞检查，
17.916s，零mj_step：approved-circle-standing-ik-20261108T061637Z。
左臂新目标下的整场景名义反向弧线亦无禁止接触：
fetch-chair-reverse-parked-left-static-20261108T061941Z。
这两项仍是离线预检，后续真实8秒持瓶和所有动态门必须重新通过。

### 新目标：离桌转向后，单独实测持瓶向前一米（2026-11-08）

维护者清除先前长Goal后，明确将当前目标限定为：真实抓取/抬升/收臂，
离桌并转到与桌前朝向相反的方向，再持瓶向前至少1.00m并停稳。
凳子到达/放置、四方向微调、LiDAR、语音和C++迁移均不属于这个Goal。

先复核现有接触故障，不改接触模型。原抓取截面在COM上方0.10m；
固定站位下0与0.05m未通过当前IK，反向工具轴候选也未通过。
0.075m通过；发现原规划只用URDF/model范围，而MC交集的腕pitch较窄。
新增只读规划约束交集（不写model.jnt_range），并在收臂搜索中留额外
0.03rad补偿余量。0.075m截面、[-0.15,-0.10,0]m收臂及左停放的
实际完整CLI静态预检通过，源输入/所有观测姿态保持原样：
approved-circle-standing-ik-20261108T081745Z（0物理步）。
这不等于新高度已经动态抓取/携带通过。

同一已有Micro桥、PointGoalController、唯一发布者和原复合停止合同
执行两个阶段：有界后退弧线完成约180度转向；真实停稳并核对朝向后，
才建立一米段的独立真实位姿原点。第二段按实测前向投影达到1.00m触发
置零，最终投影仍须>=1.00m、位置误差<=0.15m、横向<=0.05m、
朝向误差<=0.12rad以及完整原停止门通过。转向段的移动不计入这一米。
每段使用独立的一次性许可；全任务共享原绝对90s命令期限，不重置时钟，
中间与最终制动各仍有界<=15s，只有实际中间复合停止PASS才允许第二段。
原始单段携带配置保持兼容，时钟/物理故障不借此重武装。

393个全场景名义几何姿态检查无禁止接触；名义转向-3.06163rad，
单独一米段投影1.00053m。这仅是运动学/状态流和静态碰撞预检：
carry-away1m-full-mission-static-20261108T082039Z。
真实外层入口到原生层参数与前容器校验亦通过，未启动MC/物理：
carry-away1m-real-entry-offline-preflight-20261108T082323Z。

### 空瓶允许倾斜：任务口径修正（维护者直接澄清）

083912Z真实新高度抓取/抬升/收臂通过；离桌约1.735s后，原竖直倾角门
停止，1m段NOT_RUN。末端手轴倾角0.03382rad，瓶轴0.12744rad，
相对轴角0.11482rad，抓取点误差3.059mm；旧FAIL原样保留。
原生夹爪ctrl约1.27s起已到0，力矩约0.8Nm，2Nm上限未被用满；
这是可检验的控制余量问题，不据此宣称已需要改接触模型。

维护者明确“瓶子完全可以倾斜，里面没有水”。新显式
--allow-carried-bottle-tilt只在合格抬升后的CARRY_HOLD取消世界竖直
0.10rad硬失败，继续记录实际倾角。没有修改物理参数、网格或控制目标；
本轮也不同时增加kp。原10mm抓取点位移、双侧失联0.10s、
掉落/外部支撑/碰撞、机器人姿态与源限制及完整停止门均保留。
初始相机的竖直物体假设与闭合前目标检查不变。
审批口径：carry-away1m-empty-bottle-clarification-20261108T085656Z。


## 2026-11-10: H220 lift and released-table manifold diagnosis

Actual episode035240Z completed H220 fresh RGB-D postured grasp/lift:
1.263mm circle arrival error, minimum58.770mm lift,1.0 simulated second
unsupported hold, zero table force, minimum pad normals15.121/14.695N.
The controlled4027-frame source/posture audit passed, maximum root planar
speed0.072675m/s and claw torque1.86373Nm. This does not qualify H220 transfer.

H300 full START episode041253Z physically completed both-table transfer and
independent table support2.0s. Its original overall failure,
CARRY_OWNER_TERMINATION_UNCONFIRMED, is preserved: the owner exit0 appeared
after the bounded parent closeout. The postmortem physical result is separate.
Carry stdout now contains a bounded preview while the complete trace remains
in native_hal_pregrasp_trace.json; no shutdown success is inferred from a marker.

Episodes042535Z and044427Z again arrived/released, but did not pass the
unchanged continuous2-second support stability gate. The former recorded only
1.94s after angular speed finally fell below0.1rad/s; the latter still rocked
above0.1 within the original35s total placement budget. Merely waiting longer
was not established as a fix. Their failures remain unchanged.

All1105 post-placement native samples in044427Z had exactly ONE
bottle/delivery-table contact. Contact location moved around the bottle rim;
the base speed approximately matched angular speed times the0.15m COM height.
This supports contact-manifold rocking as a specific numerical hypothesis,
not a gain or geometry diagnosis.
[MuJoCo3.3.7 documents](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option-flag-multiccd)
single-point flat-contact instability and the optional multiccd algorithm.

An isolated real-physics comparison used the EXACT installed3.3.7 library
SHA25696af6d24fe6d75ce16377bf4da7a8d7481274e0f86d4278ef6b6e0cb61d54eb4,
verified as the sole loaded MuJoCo library. It used the same full scene and
pre-step bottle initialization above the second table, no MC and no robot
control. Five simulated seconds per case changed only enableflags0->16.
Contact count changed1->5; last-two-second maximum angular speed changed
0.061382->0.002464rad/s, planar speed0.009128->0.000369m/s. No other bottle
interaction occurred. Both isolated cases passed; this comparison does NOT
prove the full MC release sequence or authorize a task PASS.
Evidence: reviews/issue-46/native337-bottle-table-multiccd-comparison-20261110T050856Z/.

The new --native-multiccd / review-driver --multiccd is a separately labeled
SIMULATION_ONLY numerical comparison. It preserves the old default profile,
all427 compiled arrays, mesh/assembly/material/mass/inertia/gains and source
joint/actuator limits. Actual loaded native flags must be observed, and a new
contact-free/opening/grasp validation runs once for this changed collision
algorithm. No old cached grasp acceptance is reused. No geometry/support is
added, and the original material,2Nm limit and stop criteria remain unchanged.
Full MC dynamics under that opt-in profile are still pending.
