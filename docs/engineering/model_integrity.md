# Robot Model Integrity

This is a mandatory, high-priority RobotSim contract for vendor robot and gripper models. It applies to production assets, adapters, conversions, test fixtures, and `SIMULATION_ONLY` research. A simulation-only label narrows a claim; it does not grant permission to invent a mechanism.

## A. Official-source-first

- Prefer the authoritative vendor robot description and original geometry for the exact robot/version being modeled.
- Record the upstream repository, immutable commit or release, source file path, relevant SHA-256 values, and applicable license/provenance notes.
- A repository-level license does not automatically establish redistribution rights for every model or mesh file.
- Never silently substitute generated geometry, a different product revision, or another robot's mechanism for official assets.

## B. No unauthorized mechanical reconstruction

Without explicit, case-specific maintainer approval, agents must not create, invent, simplify, or reconstruct mechanical model behavior. Prohibited changes include:

- replacing an official hand or gripper with a custom mechanism;
- inventing links, fingers, hinges, bearings, axes, joint origins, limits, or assembly placements;
- changing a fixed joint to revolute or vice versa;
- changing linkage, mimic, gear, spring, tendon, or closed-loop relationships;
- adding equality, weld, or loop constraints not established by an authoritative source;
- changing collision geometry, contact exclusions, or collision masks to make a grasp or simulation pass;
- presenting an experimental equivalent mechanism as the manufacturer's model.

These restrictions also apply to offline experiments and research branches.

## C. Case-specific human approval

Before implementing any mechanical-model change, stop and present the maintainer with:

1. the exact source files and pinned upstream revision;
2. the observed defect or missing information, with reproducible evidence;
3. the proposed delta and exact affected links, joints, transforms, geometry, and constraints;
4. authoritative mechanical/vendor documentation that supports the delta;
5. expected kinematic, collision, and dynamics effects; and
6. the validation plan and claim boundary.

Proceed only after explicit approval for that specific proposal. General instructions to fix a gripper, improve contact, or complete a grasp do not approve mechanical redesign. Mechanical-model changes are high-risk and are never an automatic self-merge case.

## C.1 Standing authorization: Issue #46 OmniPicker simulation parameter debugging

**Maintainer clarification, 2026-11-06.** This authorization persists across
chats working on Issue #46; read it before requesting another parameter approval.
The protected artifact is the human-reviewed OmniPicker STL shape, assembly and
0–100% kinematic motion. Ordinary simulation control/parameter debugging that
preserves those artifacts is not a request to reconstruct the mechanism.

Within the existing Issue #46 research worktree and separate `SIMULATION_ONLY`
overlays, the maintainer authorizes bounded, evidence-based tuning of gripper
position-control gains and joint-internal friction/damping without per-parameter
confirmation. Specifically, an isolated right-claw six-joint comparison of
`frictionloss=0.3 -> 0` is authorized. Zero is a diagnostic assumption, not a
manufacturer friction specification. This supersedes the earlier pending
approval for that comparison and the gain-only proposal in the research report.
It does not automatically authorize every proposed parameter value.

Execution requirements:

- Preserve immutable upstream files and the reviewed baseline; record exact
  before/after values, input hashes, purpose and a reproducible command.
- Change one diagnosed condition at a time. A new gripper gain or
  joint-resistance configuration needs actual contact-free opening/closing
  with fresh feedback and real physics before its first contact trial.
  Reuse that result for the unchanged model/build/parameter combination;
  an unrelated path or observer edit need not repeat that cycle.
  Verify the active configuration and current required hand readiness instead.
- Keep existing force/torque, velocity, joint, posture and stop limits. This
  authorization does not permit raising limits, changing MC gains/binaries,
  weakening success criteria, or connecting hardware.
- Joint `frictionloss` is distinct from pad/object contact friction. Contact
  friction, mass, inertia, gravity and collision parameters are not covered by
  this standing authorization.
- If motion becomes unstable or a change affects the reviewed shape, assembly,
  kinematic envelope or contact geometry, stop that experiment and present the
  exact affected model for human review. Web overlays can verify appearance;
  they cannot replace dynamic/force validation.
- Do not ask again merely because a permitted control parameter affects
  dynamics. Missing real friction data does not prevent a labeled diagnostic
  comparison, but its result cannot be promoted to hardware fidelity.

An accepted visual review persists for the identified geometry/contact region
and grasp-frame version. Routine authorized parameter debugging does not reopen
an unchanged visual scope. A materially changed region, assembly, collider or
task frame must be shown explicitly. Reuse the existing web viewer instead of
building another review system or rerunning unchanged geometry fitting.

Sections B/C still require case-specific approval for geometry, body/link/joint
identity, joint types/origins/axes/limits, topology, mimic/equality/transmission
relationships and collision changes. Replacing the hand, removing contacts,
adding supports/welds or writing object poses to manufacture a grasp remain
forbidden. Canonical model promotion/merge review remains separate.

## C.2 Case authorization: local OmniPicker-bottle friction-reference diagnostic

**Direct maintainer approval, 2026-11-08.** The user approved exactly this
SIMULATION_ONLY comparison: 22 already collision-enabled source-CoACD pad/bottle
geom pairs, friction-reference time constant inherited0.02s ->0.002s, with
globalNoSlip disabled. The active native timestep is0.001s; do not silently
change timestep or extend this value to other timesteps/pairs. Explicit pairs
must preserve their previous condim, friction coefficients, normal solref/
solimp, margin/gap, all geometry/assembly/mass/inertia, and the2Nm actuator
and original velocity/posture/stop limits. No source asset edits or attachments.

Validate source/derived compiled equivalence and actual native pair parameters
before contact; qualify real static holding before the existing bounded chair
carry diagnostic. This is a numerical comparison, not manufacturer material
identification or hardware approval. Keep the original successes and failures.
This specific approval is separate from C.1 and does not authorize other contact
parameter changes or promotion/merge.

## C.3 Case authorization: bounded carry-transit root-speed diagnostic

**Direct maintainer approval, 2026-11-08.** The human approved the separate
SIMULATION_ONLY strategy: instantaneous pelvis planar speed <=0.50m/s only
during navigation and bounded braking. Command caps remain <=0.12m/s forward
and <=0.10rad/s angular; startup/standing retain the0.20m/s gate. All source
joint limits/speeds, posture, grip, collision, freshness, watchdog and complete
physical/controller-target stop gates remain unchanged.

The explicit carry profile (existing native STEP or FoundationWalk mode) is
PID/start-token and Reset-generation bound. One nonzero interval is capped at90s; braking permission lasts at most15s
and cannot extend the original absolute deadline. Revocation is irreversible;
this permit cannot authorize rearm. Whole-record audits apply the same phase
bounds. Missing, stale or invalid identity cannot enable the higher limit.
No robot geometry, MC/policy/gains, joint limits or hardware changes are approved.

The later direct empty-bottle clarification allows object-world tilt after
qualified lift; it is a task-acceptance change, not a contact/model change.
Robot posture and retention/drop/collision guards remain unchanged; see the
[empty-bottle transport criterion](../workflows/06_X2_AUTONOMY_TASK.md).


This is a separately reported simulation diagnostic, not vendor acceptance.
It does not reclassify prior strict0.20 failures as passes.

The maintainer's later explicit leave/turn-then-forward1m Goal permits two
sequential task segments. The second uses a distinct one-use permit only after
a real accepted intermediate composite stop and fresh state/ownership checks.
The original absolute90s command deadline is shared and never reset; limits
are not multiplied per segment. Permits cannot overlap or reuse a revoked
record, and whole-record audits bind both intervals to the verified arm owner.
This does not authorize recovery/rearm after a physical or clock safety fault.


The later in-place-turn request is implemented as an explicit outer-loop
ground-truth centre-hold diagnostic: only during TURN, body-frame XY correction
has total commanded magnitude <=0.12m/s and angular magnitude <=0.10rad/s.
The native MC still owns alternating leg support; no direct leg trajectories,
MC gain/policy edits or new velocity publisher are introduced. This does not
qualify general four-direction micro-motion. Original robot/source and stop
guards remain, including measured centre/yaw acceptance.

## C.4 Empty-bottle carry acceptance (2026-11-08)

The user explicitly accepts bottle wobble and relative circle/section-center
drift during carry provided the bottle is not lost or dropped. This supersedes
the10mm relative-grasp-point stop criterion only AFTER actual grasp, lift and
retract have qualified. Record that drift and tilt; do not move the circle or
object to conceal it. The separately named retention-only profile may continue
with one loaded pad; losing all pad support for the existing0.10s interval
still fails as loss of grasp. This is not proof of floor impact.

Initial approach/closure/lift/retract criteria, actual contact physics,2Nm cap,
robot/source joint limits, posture, collision, freshness and complete physical
stop remain. Do not modify model geometry, force limits or the MC to obtain a
PASS. The user-approved two-table layout and10cm fixed-stance target update
are derived episode-boundary scene changes, not an active-robot teleport.

## C.5 One in-place centre-recovery020 experiment (2026-11-08)

The user's direct reply explicitly approves ONE independent simulation:
only TURN position recovery may command one body-frame XY axis up to0.20m/s,
with angular command zero during that recovery. Normal retreat/travel stays
<=0.12m/s, nominal turn angular<=0.10rad/s, actual root<=0.50m/s.
All source-joint, posture, collision, freshness and full-stop gates remain.
No model, MC policy/gains or grip-force change is authorized by this exception.
Use an explicit opt-in; defaults remain0.12. No automatic second launch or
motion retry is covered by this one-attempt authorization. Preserve the exact
command, source revision and actual result even when it stops before motion.

## C.6 Autonomous carry-control iteration (2026-11-09)

The user's direct current instruction supersedes C.5's single-run budget for
subsequent research: autonomously choose ordinary numerical/control experiments
to complete retreat, bounded small-radius180deg turn and placement on the
approved second table. Record .20m/s SDK-amplitude transit/correction as a
separate explicit simulation profile; defaults stay unchanged. No per-run
confirmation for these routine choices. The separately named compact-turn
comparison uses pure yaw <=.60rad/s (native SDK keyboard bound is1.0), no
commanded XY during TURN, measured active radius <=.30m and settled centre
<=.20m. This is the user-permitted small-radius task, distinct from the legacy
strict50mm in-place diagnostic; no source joint or physical stop limit changes. Physical faults stop the current
episode; analyze them before another distinct attempt. This does not authorize
a different robot/hand, mechanics, collision removal, hidden support, object
attachment, state teleportation, vendor-binary edits or hardware execution.
Keep source joint limits, the existing .50m/s active-transit root bound,
posture/freshness/watchdog and the complete independent physical stop.

The compact simulation path may recover an independently measured host-wall
clock epoch only through immediate zero, >=1.05s zero publishing, accepted SD,
fresh new-epoch telemetry and the unchanged complete composite stop. Up to
four such clock recoveries share the original90s absolute command deadline;
up to six non-overlapping PID/generation-bound permits cover the original
two segments plus those recoveries. No permit is reactivated or overwritten.
The opt-in source-release stop protocol sends documented INPUTACTION_DISABLE
(2002) only for the episode-owned robotsim_official_cmd_vel after initiating
zero, retains >=1.05s zero publication and the full SD/physical stop, then sends
INPUTACTION_ENABLE (2001) before a qualified next segment. Service acceptance
is not task completion or a zero-command acknowledgement. Record actual source,
model input and physical behavior independently; no built-in source changes.

Physical, posture, source-joint, collision, stale-native or ownership faults
remain non-recoverable within the episode. The temporary timesyncd comparison
did not establish stable WSL wall time; the service was restored active.

The separately named two-table transfer task may continue from an actually
qualified stop inside its <=0.30m turn/braking footprint, using the measured
new pose, rather than demanding the legacy0.20m settled-centre precision.
The legacy result and profile remain unchanged. Three normal segments plus
up to four clock recoveries use at most seven non-overlapping one-use permits,
all sharing the original90s absolute command deadline. Table placement uses
the same sole HAL arm/hand owner; only after arrival/full stop/current SD may
the held bottle transfer contact to the approved delivery table. Intended
bottle-table support is a task-contact permission, not a collision filter or
geometry edit. Released placement requires actual table load, absent pad load,
fresh stable object motion and a complete measured support interval. No welds,
object pose writes, force-cap changes or fabricated success.

## D. Stop when source semantics are missing

If source geometry, assembly, loop semantics, joint type, pin pairing, collision policy, or physical parameters are insufficient, stop before building a guessed mechanism. The explicitly authorized simulation parameter diagnostics in C.1 are an exception for those parameters only; they do not establish missing mechanical semantics. Record the exact missing information and request manufacturer/maintainer clarification. Do not tune guessed parameters or repeat speculative variants to force a result.

## E. Immutable upstream and source-preserving adaptations

- Treat fetched vendor files as immutable. Put an authorized adaptation in a separate, reviewable overlay and retain the exact upstream-to-adaptation diff.
- A conversion is allowed only when its source revision, units, transforms, axes, limits, inertias, geometry mapping, and constraint semantics are understood and checked.
- Preserve unsupported or custom upstream tags as unresolved; do not silently choose their mechanical meaning.
- Keep source asset identity separate from simulator-only actuators, fixed-pose constraints, collision filters, fixtures, and controller assumptions.

## F. No automatic model promotion

Research and generated models must not become canonical robot assets, production runtime dependencies, or default integration models without explicit maintainer approval. A successful simulation is evidence only for the measured setup; it does not prove source fidelity or hardware authenticity.

Classify model inputs and outputs as `OFFICIAL_UPSTREAM`, `VERIFIED_SOURCE_PRESERVING_ADAPTATION`, `UNVERIFIED_EXPERIMENTAL_MODEL`, `GENERATED_EVIDENCE`, or `UNKNOWN_REQUIRES_HUMAN_REVIEW`. Unknown provenance remains unknown. Keep historical evidence, but isolate unverified experimental models from production and runtime.

## G. Enforcement boundary

`scripts/agent/check_governance.py` verifies that this contract remains linked from root `AGENTS.md` and rejects direct runtime-source references to `scripts/research/` or local `reviews/` evidence. This static check does not prevent an agent from editing a model file, validate a claimed approval, or establish vendor fidelity. Human review remains required. Stronger enforcement through `CODEOWNERS` and branch protection is a separate repository-settings decision; it is not implied by this document or by passing CI.

The current Issue #46 inventory is a review-only [model-integrity and cleanup manifest](../research/issue46_omnipicker_model_integrity_cleanup_manifest.md). Its proposed dispositions do not authorize cleanup.
