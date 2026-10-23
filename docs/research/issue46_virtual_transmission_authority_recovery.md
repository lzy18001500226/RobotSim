# Issue #46: Virtual-Transmission Controller Authority Recovery

**Result: FAIL — the Stage 3 operational OPEN hold crossed source hard limits and velocity limits at 0.004 s.** The virtual-transmission architecture is not rejected by this result. The tested bias-aware controller profile is not stable for the full hand, so no later prevalidation or bottle stage was run.

## Scope and Runtime

This recovery tested the source-bounded control-authority question, one bias-feedforward sign check, and the required clean-reset 2 s operational OPEN hold. The hold stopped after two MuJoCo steps. It had no bottle, external object, arm target movement, or self-contact. MuJoCo was 3.3.6 at a 0.002 s timestep, using Python 3.10.12 and the pinned AgiBot X2 source checkout `575cc6b988f976c23550e0db85aa1e5475d3652d`. The hand contained 20 public active coordinates and 12 simulation-internal follower coordinates.

Source joint ranges and effort limits were not changed. The operational target inset is a simulation/control margin, not a modified hardware limit. The controller is labeled **CONTROL-LEVEL / ACTUATED MIMIC EMULATION**; followers are commanded only from source mimic relations and remain internal implementation details.

## Stage 1: Authority Audit

The complete per-joint table is preserved in `controller_authority_audit.json` in the evidence directory. Its 32 rows include source range and effort, operational OPEN target, `qfrc_bias`, actuator bounds, current gains, torque at 0 / 0.0015 / 0.003 rad error, minimum `kp` needed to counter bias at 0.0015 rad, source-effort feasibility, and effective inertia. This audit was taken against the pre-change hand servo at base HEAD `e74b554`; Stage 2 and Stage 3 use the bias-aware motor implementation recorded in this report.

All 32 source effort limits are 33.5 N·m. The pre-change controller used `kp=0.02 N·m/rad`, `kv=0.0006 N·m·s/rad`, and a 0.05 N·m simulation torque cap. Its output at errors 0 / 0.0015 / 0.003 rad was 0 / 0.00003 / 0.00006 N·m. The maximum measured OPEN bias was 0.0395441485 N·m at `L_thumb_roll_joint`; the corresponding minimum `kp` to counter that bias at 0.0015 rad is 26.3627654 N·m/rad. The bias is below both the 0.05 N·m simulation cap and the 33.5 N·m source effort limit.

**Authority answer:** at the measured initialized OPEN pose, the available bounded actuator authority is sufficient to counter the static generalized bias without exceeding source effort limits. This is only a static authority result; it did not establish dynamic stability.

## Stage 2: Operational OPEN and Controller

The source hard limits remain intact. Public active targets use the source range inset by 0.0015 rad on each side. Each follower target is derived from its source mimic multiplier and offset, then checked against the same operational soft range. Initial active and follower qpos assignments occurred before `t=0` only. The run recorded 54 initialization assignments; active and follower rollout qpos writes were both zero.

The single simulation-derived profile was:

```text
tau = clamp(qfrc_bias + kp * (target - qpos) - kd * qvel,
            -min(simulation torque cap, source effort limit),
            +min(simulation torque cap, source effort limit))
kp = (effective effort bound - abs(open qfrc_bias)) / 0.0015 rad
kd = 2 * zeta * sqrt(effective joint inertia * kp), zeta = 1
```

Effective inertia was obtained from the MuJoCo mass matrix as `1 / (M^-1)[i,i]` at the initialized OPEN pose. These gains and the 0.05 N·m simulation cap are simulation-derived, not vendor parameters. Follower target velocity was derived from the source driver target velocity and mimic multiplier.

The feedforward sign was checked against `M*qacc + qfrc_bias = qfrc_actuator + qfrc_applied + qfrc_passive + qfrc_constraint`. At zero target error and velocity, the initialized one-step test measured zero generalized-force residual for each controlled hand DOF. After its one step, maximum mimic error was 0.001213759 rad, active target error 0.000467279 rad, and follower target error 0.000592279 rad; there were no hard-limit, velocity-limit, finite-state, self-contact, penetration, effort-bound, or 0.010 rad failures. The one-step test is not evidence that the full hand can hold for 2 s.

## Stage 3: Static OPEN Hold

The clean-reset hold was configured for 1,000 steps / 2 s and stopped on its first failure:

| Time | Result |
| --- | --- |
| 0 s | No hard-limit violations, self contacts, or penetration; initial mimic residual 0 |
| 0.002 s | No position or velocity limit violations; max `|qvel|` 0.296139 rad/s; max `|qacc|` 157.002 rad/s² |
| 0.004 s | 12 source position-limit violations and 20 source velocity-limit violations; max `|qvel|` 92.000877 rad/s; max `|qacc|` 46,944.735 rad/s² |

The reported position-limit set includes `L_thumb_mcp_joint` at +0.0118013 rad against its 0 rad upper limit. All violating joints and their positions, velocities, and source limits are retained in the result JSON and trace.

At the stopping step the state was finite and there was no self-contact or penetration. The maximum logged mimic residual was 0.250046158 rad, so both the 0.003 rad diagnostic target and the 0.010 rad manipulation abort ceiling were exceeded. Active and follower target errors reached 0.0605752 rad and 0.1835952 rad. The maximum logged feedforward, PD, and total commanded torque magnitudes were 0.0395459, 0.0243022, and 0.0395441 N·m, respectively, below the 0.05 N·m simulation cap. Source hard limits and actuator effort limits were not changed.

### Bounded First-Failure Diagnosis

The existing trace shows the same no-contact OPEN hold moving from `|qvel|max=0.296` / `|qacc|max=157.002` at step 1 to `|qvel|max=92.001` / `|qacc|max=46,944.735` at step 2. The effort bound was not exceeded, and there was no self-contact to explain the excursion. The first gate failure is the hard-limit violation at step 2; velocity violations occur on that same step.

This evidence identifies the tested per-coordinate bias-aware servo profile as dynamically unstable in the full coupled hand. Its gains were derived using individual effective inertias and a continuous-time critical-damping formula; those calculations did not demonstrate stability for the full coupled hand at the selected timestep. The trace does not isolate a more specific numerical mechanism. One trace-only diagnosis was completed; there was no second simulation, retuning, or parameter sweep.

## Conditional Stages Not Run

Stage 3 failed, so full virtual-transmission prevalidation, fixed-object contact, bottle hold, 5 mm lift, and 30 mm lift were not run. No bottle contact or grasp claim is made. Do not resume those stages from this controller profile.

## Evidence and Reproduction

Raw evidence is preserved under:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/controller-authority-recovery/`

Files:

- `controller_authority_audit.json` — all 32 Stage 1 joint rows
- `bias_aware_one_step.json` — feedforward sign and one-step result
- `static_open_hold_result.json` — hold outcome and gate summary
- `static_open_hold_trace.jsonl` — both measured steps, including joint states, efforts, residuals, and limits

SHA-256:

| File | SHA-256 |
| --- | --- |
| `controller_authority_audit.json` | `2187afc687589378f9538e64b194a6f8c400778361f87bcd58a651340171c240` |
| `bias_aware_one_step.json` | `98985e1d362310058af6f730d8170a90f849ed205729bfa7480d92ddd13cc66b` |
| `static_open_hold_result.json` | `25699d8a5bb3439ca689ae3f254534f54d4fd2381d0ce2c415215c6137a8f248` |
| `static_open_hold_trace.jsonl` | `7a3226cfded30b19a6834acaac3254d2880946335aeccc2dfcd0476222a36d17` |

The Stage 1 audit was originally recorded in the sibling `controller-authority/` folder; it is copied into the recovery folder unchanged. The output hashes are recorded in the closeout comment and can be checked with `sha256sum`.

Reproduction command for the failed Stage 3 hold (use a fresh evidence directory to preserve the collected files):

```bash
cd /tmp/robotsim-issue46-virtual-transmission-resume-20261007
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  ISSUE46_EVIDENCE_DIR=/tmp/issue46-vt-authority-repro-20261007 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u \
  scripts/research/issue46_virtual_transmission_authority_gate.py --static-open-hold
```

The scripts and this report are research artifacts on the Issue #46 review branch. PR #53 remains Draft. The next decision is maintainer review of the recorded Stage 3 failure; no architecture acceptance or bottle manipulation result is implied.
