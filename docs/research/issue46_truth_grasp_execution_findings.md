# Truth grasp execution: opening and pose findings

Status: incomplete physics qualification; no grasp or lift PASS. Model and source assets unchanged.

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
