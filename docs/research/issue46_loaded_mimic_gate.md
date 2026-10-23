# Issue #46: Physically Valid Initialization and Loaded Mimic Gate

**Result: READY FOR MAINTAINER X2 LOADED-MIMIC REVIEW.** The exact bottle-disabled manipulation approach passes the unchanged `0.003 rad` gate with no initial self-penetration, joint-limit violation, sign inversion, NaN, bottle contact, or active-rollout follower-qpos write. This does not authorize or report bottle grasping.

## A. Initialization

The prior reset put the unused left thumb DIP region about `19.1 mm` into the hip/body and also placed several finger pairs at collision. The corrected reset uses a relaxed left arm/hand pose and one-time follower initialization from the source mimic equations. Robot collision remains enabled. The right-hand changes below are limited to small documented neutral-pose corrections for its measured initial self-contact.

| Joint group | Source-coordinate initialization (rad) |
| --- | --- |
| Left arm | shoulder pitch `0`, roll `0.10`, yaw `0`; elbow `-0.15`; wrist yaw/pitch/roll `0` |
| Left hand | thumb abduction `0.35`, MCP `-0.10`; index abduction/PIP `0.10/0.05`; middle PIP `0.05`; ring abduction/PIP `-0.10/0.05`; pinky abduction/PIP `-0.16/0.05` |
| Right-hand initial collision correction | thumb MCP `0.05`; index/middle/ring/pinky PIP `0.05`; ring abduction `0.10` |

Before the first step, the actual task script recorded zero contacts and zero robot self-contact pairs, maximum initial penetration `0 m`, 63 limited joints with no source-range violations (inclusive boundary checks; minimum margin `0 rad` at source endpoints), and all 12 mimic residuals exactly `0 rad`. No collision group was globally disabled.

## B. Loaded Mimic Results

Runtime identity: pinned AgiBot source `575cc6b988f976c23550e0db85aa1e5475d3652d`; OmniHand URDF SHA-256 `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`; Python `3.10.12`; MuJoCo `3.3.6`; NumPy `1.26.4`; WSL2 x86-64; timestep `0.002 s`; `implicitfast`; Newton solver, 100 iterations, tolerance `1e-8`.

The isolated load cases used the same model and driver/follower ownership. The tested profile used equality `solref` scale `4` and mimic-driver `kp/kv` scale `2`; arm targets used 500 steps. These are simulation-derived settings, not upstream physical parameters.

| Bottle-disabled condition | Steps | Maximum mimic error | Result |
| --- | ---: | ---: | --- |
| Gravity only | 500 | `0.002979330 rad` | Pass, close to gate |
| Arm motion only, gravity/contact off | 500 | `0.001333555 rad` | Pass |
| Gravity plus arm motion | 500 | `0.002972314 rad` | Pass, close to gate |
| Valid first self-contact, then hold | 4,607 total; 500-step hold | `0.000812408 rad` | Pass; one thumb-DIP/pinky-PIP contact, persistent throughout hold, peak penetration `0.351 mm` |
| Finger-only ramp at original 500-step duration | 500 | `0.009724696 rad` | Fail; first breach `R_pinky_dip_joint`, step 4 |
| Full approach at original 500-step finger duration | 500 | `0.004356224 rad` | Fail; first breach `R_index_dip_joint`, step 7 |
| Full actual task-script approach with 1,000-step finger ramp | 1,000 | `0.002972441 rad` | Pass; arm ramp remains 500 steps, bottle collision disabled |

The exact actual-task run's peak occurred at step 3 on `L_thumb_dip_joint`; the per-relation maxima over all 1,000 steps are in the external trace. The worst residual is only `0.000027559 rad` below the required limit. Equality scale `2` failed (`0.006707861 rad`); increasing solver iterations from 100 to 200 did not change the equality-scale-4 result. The 2x slower finger target ramp, not a relaxed tolerance or follower actuation, is necessary for the final full-approach pass among tested candidates.

No source URDF axis, limit, mimic ratio, or driver/follower assignment changed. Equality scaling and actuator-gain scaling are simulation-derived. The source mimic driver remains the only actuated joint in each relation. The follower is seeded before stepping to satisfy the relation; active-rollout follower-qpos writes are zero.

## C. Gate Checks

For the exact actual task-script path: all 12 relations remained at or below `0.003 rad`; there was no first breach, no source-range violation, zero sign inversions, no NaNs, zero robot self-contact through the approach, zero bottle contacts, and zero active-rollout follower-qpos writes. The bottle geoms were collision-disabled for this diagnostic. No bottle-contact, grasp, lift, transfer, release, or placement phase ran.

Every per-step record includes all 12 relation errors. The four final per-relation maxima were:

| Relation | Max error (rad) | Step |
| --- | ---: | ---: |
| `L_thumb_pip_joint` | `0.002723707` | 2 |
| `L_thumb_dip_joint` | `0.002972441` | 3 |
| `L_index_dip_joint` | `0.000820735` | 4 |
| `L_middle_dip_joint` | `0.000828076` | 4 |
| `L_ring_dip_joint` | `0.000830608` | 4 |
| `L_pinky_dip_joint` | `0.000822724` | 4 |
| `R_thumb_pip_joint` | `0.001597744` | 522 |
| `R_thumb_dip_joint` | `0.001242459` | 521 |
| `R_index_dip_joint` | `0.002968336` | 12 |
| `R_middle_dip_joint` | `0.002609011` | 12 |
| `R_ring_dip_joint` | `0.002318720` | 13 |
| `R_pinky_dip_joint` | `0.002086953` | 13 |

## Reproduction and Evidence

Run inside the pinned WSL2 prototype environment. The command is also preserved verbatim in the external evidence directory:

```bash
MUJOCO_GL=egl ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v25 ISSUE46_FINGER_CLOSE_FRACTION=1.0 ISSUE46_GRASP_PALM_Z_OFFSET_M=0.0 ISSUE46_MIMIC_EQ_SOLREF_SCALE=4.0 ISSUE46_MIMIC_DRIVER_GAIN_SCALE=2.0 ISSUE46_APPROACH_FINGER_DURATION_STEPS=1000 ISSUE46_MIMIC_DIAGNOSTIC_ONLY=1 /tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python /tmp/robotsim-issue46-lingxi-scene-frames/scripts/research/issue46_x2_grasp.py
```

The standalone Gate B harness command and case traces are recorded with the v22 evidence. The actual task-script result is in `runtime-parity-20261006-v25/`: `mimic_runtime_gate.json`, `mimic_gate_pre_step.json`, 1,000-row `physics_contact_trace.jsonl`, `run.log`, and the exact command. Isolated condition results/logs are preserved in v22; the same two-second finger-ramp harness result is in v23; the failed equality-scale-2 comparison is in v24. All directories are under `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/`.

**Limit:** the passing residual has little margin, so the simulation-derived profile and neutral pose remain subject to maintainer review. No grasp work is included or authorized by this result.
