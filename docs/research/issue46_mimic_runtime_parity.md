# Issue #46: Mimic Runtime Parity

**Result: FAIL — the manipulation initialization/approach path still violates the unchanged `0.003 rad` gate.** This was a contact-disabled-at-the-bottle, 500-step diagnostic only. No bottle contact, grasp phase, or tuning was run.

## Runtime Comparison

The historical integrated free-space gate passed all 12 relations over 7,000 steps with maximum error `0.0002612766 rad` (see [the accepted mimic mapping report](issue46_mimic_mapping_report.md)). The current harness reproduced that path's first step at `0 rad`. The manipulation path breaches at step 1, before any bottle contact.

| Runtime property | Accepted integrated free-space PASS | Manipulation/checkpoint path |
| --- | --- | --- |
| MuJoCo | 3.3.6 | 3.3.6 |
| `nq / nv / nu / neq` | `63 / 63 / 34 / 12` | `70 / 69 / 51 / 12` |
| Timestep, integrator | `0.002 s`, implicitfast (`3`) | `0.002 s`, implicitfast (`3`) |
| Solver / iterations / tolerance | Newton (`2`) / `100` / `1e-8` | Newton (`2`) / `100` / `1e-8` |
| Gravity | `[0, 0, 0]` | `[0, 0, -9.81]` |
| Mimic equalities | 12/12 active | 12/12 active |
| Pair model fields | qpos0, range, ref, axis signs, polynomial, solref and solimp match exactly across all 12 pairs | Same |
| Mimic driver actuator assignment | Same ten named driver joints; followers have no independent actuator | Same ten named driver joints; numeric actuator IDs shift with model ordering |
| Driver gains | Left: `18 / 2.4`; right: `0.1 / 0.003` | Both hands: `0.1 / 0.003` |
| Left-hand state / target | Interior neutral pose; left drivers target that pose | All reset to qpos/target `0`; 22 of 24 hand hinges are at a range endpoint at D |
| Geom contacts at first step | All disabled | Bottle collision disabled; five robot self-contact pairs active |
| First-step controller ordering | Set controls, `mj_forward`, `mj_step` | Set controls, `mj_step`; an added pre-step `mj_forward` produced the same result |

The full per-pair A–E records include source and MuJoCo axes, refs, qpos0/ranges, qpos/qvel, source-coordinate mapping, actuator assignment/target/gains/gear/ranges, equality activation/polynomial/solref/solimp, contact list, and limit/finite-state checks.

## First Divergence

At A, reset qpos/qvel are zero in both paths. At B–D, both paths have zero equality residual; initialization does **not** leave the mimic constraint unsatisfied. The accepted path seeds the left hand to an interior state. The manipulation path leaves 22/24 hand hinges exactly on one range endpoint and sets every left-hand driver actuator to target zero.

At C/D, the manipulation pose has these five non-bottle self contacts:

- `R_ring_pip` / `R_pinky_pip`
- `L_thumb_dip` / `left_hip_roll_link`
- `L_index_pip` / `L_middle_pip`
- `L_middle_pip` / `L_ring_pip`
- `L_ring_pip` / `L_pinky_pip`

The first dynamic divergence is E, after one `mj_step`: `L_thumb_mcp_joint` has qpos `-0.0021585918100`, qvel `-1.0792959050`; its `L_thumb_pip_joint` follower has qpos `+0.0027459344479`, qvel `+1.3729672240`. The equality residual is `+0.0056168615552 rad`, above tolerance. The reset and pre-step residuals are zero. All 12 source-to-MuJoCo driver/follower axis-sign pairs match between PASS and FAIL paths; qpos/reference conversion is not the cause.

## Ablations

Each row runs the same manipulation initialization and 500-step open/pregrasp approach target sequence. The bottle geom is disabled in every row; only the named all-contact variants disable all geoms.

| Variant | First over-tolerance sample | Max error over 500 steps | Peak out-of-range hinges at a step | Result |
| --- | --- | ---: | ---: | --- |
| Baseline | Step 1, `L_thumb_pip`, `0.00561686` | `0.10346640` | 14 | Fail |
| Gravity zero only | Step 1, `L_thumb_pip`, `0.00305162` | `0.06715503` | 12 | Fail |
| All geom contacts off only | Step 2, `L_thumb_pip`, `0.00488236`; step 1 is `0.00209666` | `0.01624707` | 18 | Fail |
| Gravity zero + all geom contacts off | Step 5, `R_pinky_dip`, `0.00356984`; step 1 is `0.00027864` | `0.01599159` | 16 | Fail |
| Forward call added before baseline step 1 | Step 1, same `L_thumb_pip` residual as baseline | `0.10346640` | 14 | Fail |

Thus the active non-bottle contact set is sufficient to explain the original step-1 threshold breach: turning all geom contacts off alone moves that sample below tolerance. Gravity contributes (`gravity` alone nearly reaches the threshold; gravity plus no contacts reduces step 1 further). The full gate still fails with both removed because the manipulation right-hand preshape/approach commands diverge later, first at step 5 on `R_pinky_dip`.

The ablation disables the complete contact set, so it does not identify which of the five individual self-contact pairs contributes most to the left-thumb residual. The gravity/contact-free run retains the actual right arm and hand approach controller; the step-5 right-pinky breach was not further separated into arm motion versus finger-target motion.

## Follow-up Runtime-Parity Isolation (v11-v15)

The later accepted-pose runs isolated load sources while retaining all 12 active mimic equalities, the accepted per-side driver profile (left `18 / 2.4`, right `0.1 / 0.003`), accepted source-coordinate hand targets, and `mj_forward` before every `mj_step`. The bottle remained collision-disabled. These runs supersede the earlier attribution that treated the step-1 failure as one initialization/contact-order defect.

At B/C/D, all mimic residuals were zero and all hand joints were in range. In the frozen-arm, normal-gravity, contacts-enabled run, `mj_forward` exposed eight existing robot self-contact pairs before the first step: right index/middle, middle/ring, ring/pinky; left index/middle, middle/ring, ring/pinky; and left thumb pip/dip against `left_hip_roll_link`. The first post-step failure was `L_thumb_dip_joint`, `0.0050619401 rad` at step 1. This is before any bottle contact. An extra `mj_forward` before the step and holding the right arm/fingers open did not resolve it.

Three one-source-at-a-time ablations then showed that no single one of gravity, self-contact, or arm motion explains the gap:

| Isolated condition (other loads disabled) | First breach | Maximum error | Peak limit violations | Interpretation |
| --- | --- | ---: | ---: | --- |
| Robot self-contact retained; gravity off; arm frozen | Step 2, `L_ring_dip_joint`, `0.0077858864 rad` | `0.3415353885 rad` | 4 | Self-contact alone breaks the gate. |
| Gravity retained; all geom contacts off; arm frozen | Step 2, `L_thumb_pip_joint`, `0.0034610542 rad` | `0.0349712896 rad` | 1 | Gravity alone breaks the gate. |
| Arm follows the 500-step approach; gravity and all geom contacts off; fingers held open | Step 11, `R_index_dip_joint`, `0.0031246882 rad` | `0.0040801032 rad` | 0 | Arm motion alone breaks the gate. |
| Arm frozen; gravity and all geom contacts off; fingers held open | No breach over 500 steps | `0 rad` | 0 | This unloaded static control passes, but is not the requested approach-path gate. |

All four follow-up cases had zero NaNs, zero driver-motion-opposed observations, zero bottle contacts, and zero active-rollout follower qpos writes. The arm-motion-only result establishes that the current accepted mimic profile also misses the strict gate under the actual arm trajectory, even without gravity or any collisions. The loaded manipulation path therefore cannot be repaired by only changing the left-hand target, disabling gravity, disabling contacts, freezing the arm, or adding a pre-step forward call.

The earliest runtime configuration differences are now explicit: the accepted free-space builder runs with gravity `[0, 0, 0]`, all geom contacts disabled, and no arm trajectory; the manipulation model runs with gravity `[0, 0, -9.81]`, robot contacts enabled, and an arm trajectory. Timestep (`0.002 s`), integrator (`implicitfast`/`3`), solver (`Newton`/`2`, 100 iterations, `1e-8` tolerance), equality activation, qpos0/ranges/refs/axes, mimic polynomial, solref, solimp, driver actuator mapping, and target coordinate conversion were matched in the accepted-pose comparison. Thus there is no single reset, axis, reference, target-coordinate, or update-order discrepancy left to fix: the accepted low-gain/equality configuration passes only in the unloaded harness and is not robust to these independent runtime loads. The exact unresolved root is insufficient dynamic robustness of the current mimic coupling under the manipulation model's physical loads. No controller, equality, collision, or initialization fix was adopted because changing one source does not satisfy the gate and changing several would become unbounded tuning.

Left-hand state and target ablations do not independently recover the gate. Seeding the accepted interior pose alone gives step-1 error `0.00837668`; setting the left target to the accepted interior value alone gives `0.00625179`; doing both gives `0.00855116`. Restoring the legacy left gains alone gives `0.00422127`, and combining those gains with the accepted left state/target gives `0.00505976`; neither passes, and the legacy-gain variants have larger later maxima. The current low-gain profile is therefore applied to both hands in manipulation, while the integrated PASS applies it only to the right hand.

## Hypotheses

| Hypothesis | Finding |
| --- | --- |
| Incorrect qpos/reference conversion | Ruled out: compiled qpos0/ref and the source-coordinate mapping agree; pre-step residual is zero. |
| Axis/sign conversion | Ruled out as a PASS/FAIL model difference: all 12 driver/follower axis-sign pairs match (`+1/+1`). The separate target-side disagreement counter is recorded in JSON and is not an axis-map test. |
| Transformed joint-limit semantics | Ruled out: qpos ranges match exactly. Runtime initialization differs: 22/24 hinges start at endpoints, then violations occur after stepping. |
| Equality polynomial/reference handling | Ruled out: coefficients, refs, solref and solimp match; all pre-step residuals are zero. |
| Inconsistent actuator target coordinates | Coordinate units and driver assignments match. Left targets do differ (`0` versus accepted interior values), but restoring them alone does not fix the failure. The right-hand manipulation target trajectory introduces a later breach even with gravity and all geom contacts disabled. |
| Equality solver configuration | Ruled out as the path difference: solver options and per-equality solref/solimp match. |
| Initialization order / unsatisfied constraint | Unsatisfied initialization is ruled out by zero A–D residuals. An extra `mj_forward` immediately before the first baseline step is behaviorally identical. |

## Integrity and Reproduction

All manipulation cases disable bottle collision and report no bottle contact. The default case retains other robot self contacts; the two all-contact ablations disable every geom contact. Active-rollout follower qpos writes: `0`. No NaNs were observed. The per-step `driver_motion_opposed_to_target_observations` field counts driver qpos displacement from qpos0 opposite the target displacement from qpos0; it is a behavior diagnostic, not a source-axis sign-conversion result.

Source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`, URDF SHA-256 `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`. Python `3.10.12`, MuJoCo `3.3.6`, NumPy `1.26.4`. RobotSim HEAD at experiment: `8f0f3ecbbf9c949ec3b8743b173b96da59720a5c`.

Exact command used (the evidence directory must be empty before the run):

```bash
wsl.exe -d Ubuntu-22.04 --exec bash -lc 'export MUJOCO_GL=egl; export ISSUE46_PARITY_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v5; export ISSUE46_PARITY_APPROACH_STEPS=500; set -o pipefail; /tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python /tmp/robotsim-issue46-lingxi-scene-frames/scripts/research/issue46_mimic_runtime_parity.py 2>&1 | tee /tmp/issue46-mimic-parity-20261006-v5.run.log'
```

Full evidence: `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v5/` (`result.json`, `runtime_identity.json`, `runtime_comparison.json`, `model_profiles.json`, `step0_pre_post_trace.jsonl`, `all_12_mimic_error_trace.jsonl`, `run.log`). The final result is **FAIL**. Grasp tuning remains paused; no grasp, bottle contact, or manipulation beyond the approach diagnostic was performed.

The follow-up source checkout was RobotSim branch `codex/issue46-lingxi-scene-frames` at `b4993bd1d90b48da396c018482887d0e2fc7dc54` (research script dirty only), with pinned vendor HEAD `575cc6b988f976c23550e0db85aa1e5475d3652d`, Python `3.10.12`, MuJoCo `3.3.6`, and NumPy `1.26.4`.

Follow-up evidence is durable under:

- `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v10/`: accepted-pose, open-hand actual arm approach under normal gravity/robot contacts; first breach step 1 on `L_thumb_dip_joint`, `0.00505958 rad`.
- `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v11/`: frozen-arm accepted-pose pre/post-step trace and eight initial contact pairs.
- `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v13/`: unloaded static control (PASS for that ablation only; arm frozen).
- `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v15/`: the three isolated load-source failures, all-12 traces, model/runtime identity, and run log.

Exact follow-up command used (the v15 evidence directory was empty at invocation; use another fresh directory to repeat it):

```bash
wsl.exe -d Ubuntu-22.04 --exec bash -lc 'set -o pipefail; export MUJOCO_GL=egl; export ISSUE46_PARITY_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v15; export ISSUE46_PARITY_SCENARIOS=accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_gravity_zero,accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_arm_motion_no_contacts,accepted_hand_state_full_targets_forward_each_step_open_during_approach_no_contacts_no_gravity; /tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python /tmp/robotsim-issue46-lingxi-scene-frames/scripts/research/issue46_mimic_runtime_parity.py 2>&1 | tee /tmp/issue46-runtime-parity-v15-run.log'
```

**Closeout: FAIL — the unchanged `0.003 rad` gate is not met by the actual manipulation runtime.** Only the diagnostic harness and this research report changed; production manipulation, hand morphology, and grasp behavior were not modified. No bottle contact or grasp/lift/placement phase was run.
