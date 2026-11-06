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
55.345mm, axis error 0.136171rad; closure was not issued. During subsequent
residual compensation, pre-close bottle movement stopped the tracker/arm.
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
