# Issue 46: single-root travel calibration review

**AWAITING_HUMAN_FULL_TRAVEL_REVIEW.** An isolated experimental one-motor
open/close cycle passed. Full commercial travel, manufacturer passive fidelity
and full X2 wrist collision validity remain **NOT VERIFIED**. No external-object
contact, grasp or lift experiment was run in this iteration. PR #86 stays Draft on #84.

## Assets and representation

The [official-URDF overlay](issue46_official_urdf_pipeline.md) still owns the
source-preserving conversion. Input is AgiBot X2 v1.4.0 at
`575cc6b988f976c23550e0db85aa1e5475d3652d`, URDF SHA-256
`35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
Original STEP SHA-256 remains
`5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b`.
No manufacturer file, STL vertex/scale, root axis/range, mimic, wrist mount,
inertial value or left-hand joint was changed. The isolated model grounds only
the gripper base; it does not certify a mounted whole-X2 configuration.

The derived URDF keeps PR #86's six experimental passive `continuous` hinges,
four accepted mesh-local bore registrations and measured D-frame re-expression.
No verified hard stops exist for the passive joints. They therefore remain
continuous; assigning plausible revolute bounds would invent mechanism data.
The official root ranges remain `[-1,0]` / `[0,1]` and axes `(0,0,-1)`.

## Coordinates and actual source coverage

`q_root = -0.13412598906686654 - q_CAD`, `q_wide = -q_root`.

The registered STEP source assembly is at **q_CAD=0**. There are no measured
moving STEP poses. The saved CSV contains 33 **computed candidate** poses per
side: narrow `[-0.32,0]`, wide `[0,0.32]`. These implement an unverified relation;
they are not manufacturer measurements. All narrow q_CAD>0 poses are **analytic
continuation**, including the old manually chosen `Q_OPEN=0.2` demonstration.
That value was neither a manufacturer stop nor a measured maximum.

The official root interval maps algebraically to q_CAD approximately
`[-0.134126,+0.865874]`. This is a coordinate conversion, not reachability proof.
Both branches, loop lengths, phase continuity, D gaps, axis directions and
equality rank were checked on a 0.005 rad grid until their first stop.

## Surface-based aperture and travel boundaries

We intersect the original, registered narrow3/wide3 **STL triangle surfaces**
with a common +Y ray at gripper-base **x=0, z=115 mm**, selecting the two inner
surfaces. Additional sections cover z=110…120 mm. This is a defined physical
section clearance, not a tip distance, global minimum, largest graspable cylinder
or manufacturer aperture specification. Full-height usable grasp width remains
unverified; a missing section is reported unavailable, not inferred from tips.

| q_CAD (rad) | q_root (rad) | Reference section aperture (mm) | Meaning |
| --- | --- | --- | --- |
| −0.085 | −0.049126 | 18.147 | Last proxy-contact-free closing grid point |
| −0.075 | −0.059126 | 19.493 | Interior closing command used in the successful physical test |
| 0 | −0.134126 | 30.373 | Registered source assembly; **not fully closed** |
| +0.10 | −0.234126 | 45.154 | Analytic candidate continuation |
| +0.20 | −0.334126 | 61.077 | Largest completed stable physical command stage |
| +0.235 | −0.369126 | 66.624 | Last algebraically solvable/proxy-clear opening grid point; **not a physical full-travel PASS** |

The closing sweep stops at **q_CAD=−0.090**: middle-link / loop collision proxies
begin contacting (narrow 0.04685 mm, wide up to 0.03146 mm). Original triangle
checks find no new middle/loop surface crossing there. This is evidence of proxy
contact without a matching surface crossing, not permission to exclude it.
At the old root-zero reference q_CAD=−0.134126, original middle/loop triangles
**do** intersect on both sides. This pose is retained only as a diagnostic;
it is not included in the verified physical motion.

The opening sweep fails at q_CAD=+0.240: the **existing candidate** has no real
closure. Bisection locates its algebraic boundary near **+0.2381657674 rad**.
This does not establish the real gripper's stop or invalidate its official limits;
it limits this experimental relationship. No branch switch or added actuator was
used to continue beyond it. Largest adjacent passive-angle grid step was about
0.07816 rad near that boundary.

Original-triangle checks at **67** already bounded grid poses found no **new
nonadjacent crossing relative to the source-pose contact baseline**. Existing
parent/pin/D mating contacts are listed with triangle witnesses. They are not
silently classified as harmless volume overlaps. No exact solid-volume or
manufacturer mate certification is claimed.

## One motor, actual passive dynamics

The new MJCF has **nu=1**, driving only `right_claw_joint`. All six downstream
hinges and the opposite root are unactuated. The two D connects and official
root mimic remain. PR #84's two explicit experimental fixed-tendon relations
restore one effective system DOF (8 coordinates, equality rank 7):

`q_root + q_B − q_E = PHASE`, with the opposite sign on the wide side.

This is the same candidate relation, not a new manufacturer assertion. The old
MJCF used root `ref=±PHASE`; the URDF-generated model uses `ref=0`. The fixed
tendon therefore needs the explicit `±PHASE` constant. A regression checks this
coordinate re-expression; blindly restoring a zero-offset equality would be wrong.

Only the root receives control. There are **zero passive qpos writes after
initialization**, no follower position targets/hidden controllers, extra motors,
object attachments, welds or mocap. The old seven-servo generator is now static
reference export only; its historical physics evidence remains separate.

Root `kp=0.5`, `kv=0.05`, ±0.5 Nm cap are unchanged simulation parameters, not
CAD measurements. The uncompensated extension failed at command +0.10: actual
q reached +0.131675, exceeding the 0.02 rad endpoint gate. Projecting model
gravity/Coriolis bias through the constraint-Jacobian nullspace predicts the
observed ~0.0317 rad steady error. Root-only model-bias compensation was added
and checked against independent finite-difference potential energy. No passive
control or gain/force-limit tuning was introduced.

With that controller, commands +0.05/+0.10/+0.15/+0.20 settled correctly. The next
+0.225 attempt reached actual +0.236739, with speed 14.803 rad/s, D gap 0.20565 mm
and coupling residual 0.009348 rad, crossing the static-interval stop at step 8710.
It stopped immediately. There was no further boundary search or softness tuning.
The final cycle confirms only the last completed +0.20 stage with an interior
closing target −0.075, rather than claiming the +0.235 algebraic bound as usable.

### Successful bounded cycle

- MuJoCo **3.3.6**, 1 ms, **19,750 mj_step / 19.75 s**, gravity on throughout.
- Source → staged opening → closing → reopening → source; one root actuator.
- Actual q_CAD **−0.074995…+0.200000 rad**; root commands **−0.334126…−0.059126**.
- Max D gap **0.019109 mm**; coupling residual **0.00108672 rad**.
- Max joint speed **0.446966 rad/s**; measured root-limit overshoot **0**.
- Enabled-contact proxy penetration **0**; inherited four internal exclusions
  remain unchanged. This is not zero penetration for excluded CAD mating pairs.
- Max generalized constraint-force magnitude **0.81865**, with mixed translational
  and angular units; per-equality values are in the trace.
- Sampled saved actual states give section aperture **19.4536…61.2316 mm**, max passive
  angular deviation from the ideal candidate law **0.020046 rad**, and root
  mimic error **0.00043918 rad**. These errors are disclosed, not exact tracking.
- No object/contact/grasp/lift stage was attempted in this iteration.

## Wrist and mechanical-fidelity limits

The unchanged mount is xyz `(1.96,0.35,−62) mm`, rpy `(3.14,0,0)`.
At the source pose, the wrist visual mesh crosses only the fixed gripper-base
interface. The official wrist **collision** mesh additionally intersects moving
narrow1, narrow3, narrow_loop and wide1 geometry. Its intended fits versus
collision-proxy errors are unresolved. We stop mounted-clearance verification at
this baseline failure; the isolated rig does not silently remove or approve it.

The **maximum manufacturer/mechanically verified full travel is NOT ESTABLISHED**.
Needed evidence is the actual per-side passive-angle relationship or reference
joint poses at known open/closed command fractions, plus classification/correction
of the identified collision representations. There is no justification to extend
the current candidate to root −1, force full closure, or promote it to production.

## Evidence / reproducibility

Final local review package:
`reviews/issue-46/single-motor-full-travel-20261022T084103Z/`.
It includes derived full URDF, exact official diff, one-motor MJCF, kinematic and
actual-physics aperture CSVs, annotated **real physics** MP4, joint/contact/force
trace, triangle witnesses, hashes and artifact manifest. No geometry/video is
committed to Git. This export copied and hash-checked the sealed physical run
`083435Z`; measurements replay recorded states only for geometry queries, with
**zero new physics steps**, and do not pose as actuation.

Failures are preserved: `082633Z` uncompensated motor test (4000 steps),
`082924Z` compensated near-boundary test (8710 steps). The first bundle's **raw
surface analysis is superseded**: it incorrectly applied compiled mesh frames to
raw STL vertices. Corrected URDF-body/mesh frames are regression-checked against
MuJoCo compiled vertices; use only `082924Z` and later surface results. A measure
dispatch bug prematurely sealed `083435Z`; it was fixed with a regression, and
new export artifacts use the new bundle instead of overwriting its manifest.

```bash
cd /home/lzy18001500226/robotsim-issue46-official-urdf-articulation
PY=/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python
$PY -m unittest discover -s scripts/research -p 'test_issue46_*py' -v
MUJOCO_GL=egl $PY scripts/research/issue46_single_motor_travel.py --stage analyze
# Use the unique printed run as RUN; never reuse a completed evidence directory.
MUJOCO_GL=egl $PY scripts/research/issue46_single_motor_travel.py \
  --stage physics --root-bias-compensation --run-dir "$RUN"
# The extended test is expected to stop near the singular region. Confirm only
# its already completed stage with a newly reserved run:
MUJOCO_GL=egl $PY scripts/research/issue46_single_motor_travel.py \
  --stage confirm --from-run "$RUN"
# Native human review of the saved successful model (gravity on, one motor):
MUJOCO_GL=glfw $PY scripts/research/issue46_urdf_review.py --single-motor \
  --model-run /home/lzy18001500226/projects/RobotSim/reviews/issue-46/single-motor-full-travel-20261022T084103Z --port 8782
# Browser scalar control and actual-physics video: http://127.0.0.1:8782/
```

Use the provided launcher: it initializes the mapped STEP pose once before
stepping. Raw MJCF default zero qpos is not that assembly pose; resetting a generic
viewer to all-zero coordinates is not the reviewed initialization.

Hosted CI is static/infrastructure evidence; it does not run these private-asset
MuJoCo experiments. Keep the four exclusions, source-fidelity limitations and
gravity compensation visible in review. Only the maintainer can authorize a
subsequent bottle-contact phase. Research disposition: KEEP-AS-TOOL pending review;
no production/runtime source depends on this experiment.
