# Issue #46: OmniHand Mimic Mapping and Runtime Coupling

**Bounded result: PASS for the free-space mimic-coupling gate. Physical grasp status remains FAIL and was not rerun.** The pinky anomaly is not a URDF-to-MuJoCo coordinate sign/reference conversion error. It is a runtime actuator/equality coupling failure under the legacy finger-driver servo gains.

## Scope

This work isolates `R_pinky_pip_joint -> R_pinky_dip_joint`, then checks all 12 pinned URDF mimic relations through a hand-only sweep and a free-space sweep in the accepted X2 robot model. It contains no bottle, table, environment contacts, or grasp trajectory. The accepted morphology, hand visuals, floor, table, bottle, station, and walking behavior are unchanged. The final physical grasp attempt remains a historical failure; no manipulation rollout was run after this coupling correction.

The existing research grasp harness now uses the measured right-hand mimic-driver gains and direct hand-joint limit solver profile. The grasp trajectory and scene were not changed. That harness edit has not been exercised in a bottle rollout.

## Runtime and Source

| Item | Verified value |
| --- | --- |
| Vendor source | `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d` |
| Model | `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf` |
| URDF SHA-256 | `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259` |
| Vendor checkout | Exact pinned commit; clean worktree |
| Host | WSL2, Linux `6.6.87.2-microsoft-standard-WSL2`, Ubuntu 22.04 |
| Python / NumPy | Python `3.10.12`; NumPy `1.26.4` |
| MuJoCo | `3.3.6` |
| MuJoCo timestep | `0.002 s` |
| Rendering | `MUJOCO_GL=egl`; H.264 video, `1280x720`, 15 fps, 89 frames, 5.93 s |
| Evidence run source HEAD | `c91b940ef8b07e81cb9a98bea711bd8ceb77f778` on `codex/issue46-lingxi-scene-frames`; the diagnostic harness was untracked at experiment time and is included in this follow-up change |
| Diagnostic harness SHA-256 | `120718c0dbf5992e3378d96fff49776dee5aa38bf8e6b762de924872e0b9c193` |

## Coordinate Mapping

For a URDF revolute coordinate `theta_src`, the diagnostic uses:

```text
theta_src = theta_src_neutral
          + sign(dot(axis_urdf, axis_mujoco)) * (qpos - qpos0)
```

For the audited right-hand hinges, `theta_src_neutral = 0`, `MjSpec.joint.ref = 0`, `model.qpos0 = 0`, and the imported MuJoCo axis equals the URDF axis (`sign = +1`). Therefore `theta_src = qpos` for `R_pinky_pip_joint` and `R_pinky_dip_joint`; their physical displacement is also `qpos - qpos0 = qpos`. This conclusion is based on the compiled model, not an assumption that raw qpos always equals a URDF angle.

| Joint | Source axis | Source lower / upper (rad) | MuJoCo axis | `MjSpec.ref` | `qpos0` | MuJoCo range (rad) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `R_pinky_pip_joint` | `[0, 1, 0]` | `[0, 1.570796326795]` | `[0, 1, 0]` | `0` | `0` | `[0, 1.570796326795]` |
| `R_pinky_dip_joint` | `[0, 1, 0]` | `[0, 1.832595714594]` | `[0, 1, 0]` | `0` | `0` | `[0, 1.832595714594]` |

The actual pair equation is:

```text
(qpos_dip - qpos0_dip) = 0 + 1.097 * (qpos_pip - qpos0_pip)
```

The compiled `polycoef` is `[0, 1.097, 0, 0, 0]`, matching the pinned URDF multiplier `1.097` and offset `0`. At reset both the custom residual and MuJoCo EFC equality residual are zero. Across the pair profiles, the maximum difference between the diagnostic formula and MuJoCo's EFC equality residual is `0.0 rad`. Per-step traces record source limits and axes, MuJoCo axes/ranges and refs, raw qpos, `qpos-qpos0`, target coordinate, expected follower coordinate, formula residual, and EFC residual.

All 12 source mimic relations compile with matching driver/follower names, multiplier, offset, and polynomial coefficients:

| Driver | Follower | Multiplier | Offset |
| --- | --- | ---: | ---: |
| `L_thumb_mcp_joint` | `L_thumb_pip_joint` | 1.33 | 0 |
| `L_thumb_mcp_joint` | `L_thumb_dip_joint` | 1.30 | 0 |
| `L_index_pip_joint` | `L_index_dip_joint` | 1.097 | 0 |
| `L_middle_pip_joint` | `L_middle_dip_joint` | 1.097 | 0 |
| `L_ring_pip_joint` | `L_ring_dip_joint` | 1.097 | 0 |
| `L_pinky_pip_joint` | `L_pinky_dip_joint` | 1.097 | 0 |
| `R_thumb_mcp_joint` | `R_thumb_pip_joint` | 1.33 | 0 |
| `R_thumb_mcp_joint` | `R_thumb_dip_joint` | 1.30 | 0 |
| `R_index_pip_joint` | `R_index_dip_joint` | 1.097 | 0 |
| `R_middle_pip_joint` | `R_middle_dip_joint` | 1.097 | 0 |
| `R_ring_pip_joint` | `R_ring_dip_joint` | 1.097 | 0 |
| `R_pinky_pip_joint` | `R_pinky_dip_joint` | 1.097 | 0 |

## Root Cause

**Primary classification: D. actuator/controller mismatch.** The coordinate conversion, axis signs, and equality coefficients are correct. The legacy independent position servos on the mimic drivers use `kp=18.0, kv=2.4`. In a low-inertia hand, this command profile excites the driver/follower system faster than the configured soft equality constraint keeps the follower aligned. The source joint range is also enforced by MuJoCo constraints rather than by rewriting/clamping qpos, so an unstable runtime state can leave the compiled range; such a negative qpos is not a hidden coordinate remap.

In the recorded physical failure, mimic error was already `0.0118437304 rad` at step 1 (`0.002 s`), while first bottle contact was at `2.672 s`. At step 1060 (`2.120 s`), `R_pinky_dip_joint` was `-1.8865687369 rad`, outside its source range `[0, 1.8325957146]`; the source-coordinate relation expected approximately `+1.227 rad`. Since the tested mapping for this joint is identity, that value is a real negative joint coordinate. Contact and grasp trajectory forces cannot explain the step-1 failure and are not the cause of this coupling discrepancy.

The single-pair profile comparison isolates the correction. Each profile runs the same 7,000-step, 14-second open/partial/full/open driver sweep with no other finger or contact geometry:

| Equality solver | Driver servo `kp / kv` | Max relation error (rad) | Limit violation | NaN / sign inversion | Gate |
| --- | ---: | ---: | ---: | --- | --- |
| MuJoCo default | `18.0 / 2.4` | `0.6598688878` | 0 | no / no | fail |
| Direct `[-10000, -200]` | `18.0 / 2.4` | `0.2333894654` | 0 | no / no | fail |
| MuJoCo default | `0.01 / 0.001` | `0.0001208375` | 0 | no / no | pass |
| Direct `[-10000, -200]` | `0.01 / 0.001` | `0.0000294112` | 0 | no / no | pass |
| Direct `[-10000, -200]` | `0.1 / 0.003` | `0.0002734876` | 0 | no / no | pass |

The driver reached `1.5707963268 rad` and the follower reached `1.7231635705 rad` in the passing moderate profile, consistent with `1.097 * driver`. Reducing driver gain is necessary in this comparison; switching the equality from default to direct format alone does not pass. The `0.1 / 0.003` profile was selected for the multi-pair and integrated checks. The runtime does not make the equality mathematically exact at every instant; the measured error remains below the unchanged `0.003 rad` acceptance gate.

## Hypothesis Results

| Hypothesis | Finding |
| --- | --- |
| Incorrect qpos/reference conversion | Ruled out for the pinky pair. `MjSpec.ref=qpos0=0`; the source and compiled axes match; the mapping is identity. |
| Axis/sign conversion | Ruled out for audited right-hand hinge joints. Source and compiled axes align, and isolated index abduction and thumb-MCP commands moved in the commanded source direction. |
| Transformed joint-limit semantics | No transformed range was found: compiled `jnt_range` equals the URDF range. MuJoCo's limit is a solver constraint, not a hard qpos clamp; the passing profile applies the tested direct limit solver to hand hinges and records zero excursions. |
| Equality polynomial/reference handling | Ruled out for the tested model. The `qpos-qpos0` polynomial matches the URDF relation; reset residual is zero and the computed residual matches MuJoCo EFC exactly. |
| Inconsistent actuator target coordinates | Ruled out for the audited coordinates: actuator target conversion uses the same source-to-qpos map. The tested controller gain, rather than its target coordinate, explains the profile split. |
| Equality solver configuration | Contributes but is not sufficient as a fix. Direct equality with legacy gains still fails; default equality with low gains passes the isolated pair. |
| Initialization violates constraint | Ruled out for the pair reset: initial formula and EFC residual are both zero. The observed physical failure starts after control begins, before contact. |

## Hand-Only and Integrated Results

The selected profile uses direct equality `solref=[-10000, -200]` and the same direct-format hand joint limit solver. In the all-12 hand-only sweep, both hands' mimic drivers use `kp=0.1, kv=0.003`, while nonmimic hold axes use `kp=18, kv=2.4`. In the integrated right-hand-only sweep, the right mimic drivers use `kp=0.1, kv=0.003`; left mimic drivers and nonmimic axes are held at neutral with `kp=18, kv=2.4`. The full source hand is swept open, partial-close, full-close, and open for 7,000 steps / 14 simulated seconds. The rendered MP4 shows the pinned hand geometry through the open/closed sweep.

| Check | All-12 hand model | Integrated X2 model, free space |
| --- | ---: | ---: |
| Source mimic relations | 12 | 12 |
| Steps / simulated time | 7,000 / 14.0 s | 7,000 / 14.0 s |
| Maximum runtime relation error | `0.0002611121 rad` | `0.0002612766 rad` |
| Gate | `0.003 rad` | `0.003 rad` |
| Maximum joint-limit violation | `0 rad` | `0 rad` |
| NaN / sign inversion | no / no | no / no |
| Contacts | 0 | 0 |
| Active-rollout qpos writes | 0 | 0 |

The initial open pose was seeded once before each rollout so the complete starting hand state satisfied the relations. No qpos values were written after rollout start, and no follower qpos was forced during motion. In the integrated X2 check, right-hand driver targets performed the sweep while the left hand was held at neutral. This validates only free-space coupling and motion; it does not validate contact loading, grip strength, or bottle manipulation.

## Corrective Harness Change

The research grasp harness now gives right-hand joints that drive URDF mimic followers the validated `kp=0.1, kv=0.003` position-servo profile. Left-hand mimic-driver holds and nonmimic finger joints retain `kp=18, kv=2.4`; mimic followers still have no independent actuator. All source hand hinge limits use direct `solref=[-10000, -200]`, matching the tested integrated profile. This is a runtime coupling correction only: the URDF, scene, trajectory, bottle, and acceptance thresholds are unchanged. It has not been run with bottle contact after the change.

## Evidence and Reproduction

Persistent attempt 17 evidence directory:

```text
/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/attempts/20261005-mimic-coordinate-runtime-coupling-17/
```

It contains `runtime_identity.json`, `pair_summary.json`, `single_pair_trace.jsonl`, `all_12_summary.json`, `all_12_mimic_trace.jsonl`, `integrated_x2_summary.json`, `integrated_x2_free_space_trace.jsonl`, `experiment_status.json`, `hand_only.mp4`, `hand_only_open.png`, and `hand_only_closed.png`. The raw traces and media are external evidence and are not committed to Git.

SHA-256 for the principal raw traces and visual artifacts:

| File | SHA-256 |
| --- | --- |
| `single_pair_trace.jsonl` | `b8f603d4bb56361537006a332a5a414847d0ea3c45d30f7903c3f8753e497223` |
| `all_12_mimic_trace.jsonl` | `6e574089decee10de35dde3a1aa5433d73066f5dfc223597233e9516495f2e69` |
| `integrated_x2_free_space_trace.jsonl` | `6abe016acd3a7dc4d0eea2e34edc9a7b9ca0614095aa2259c42c057d8d4a52e7` |
| `hand_only.mp4` | `38cd7438eb72c3e1a326101f86d636274756ffbd428faddf86c4d3865bfe228d` |
| `hand_only_open.png` | `b95021c3480b7ec9e07277d15d8e43bb1f219e5379ce4545b8b5f213ce3eaf71` |
| `hand_only_closed.png` | `35bc332ad675f82726022d2b8d09d7f75ee83dc9a7e792499aceb133cb8ce4c1` |

From the RobotSim repository root, the exact diagnostic reproduction command is:

```bash
evidence_dir=$(mktemp -d /tmp/issue46-mimic-diagnostic.XXXXXX)
MUJOCO_GL=egl \
AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-agibot-x2-urdf-575cc6b988f976c23550e0db85aa1e5475d3652d \
ISSUE46_EVIDENCE_DIR="$evidence_dir" \
/tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python \
scripts/research/issue46_mimic_coordinate_diagnostic.py
```

The command runs only the pair profiles, hand-only sweep, and integrated X2 free-space sweep. It does not run the bottle/grasp harness. The corrective harness edit remains unverified against a physical rollout by design.

## Conclusion

**PASS for the requested mimic-mapping and free-space coupling gate.** The negative pinky coordinate is explained as an actual out-of-range MuJoCo state caused by the legacy high-gain driver/equality dynamics, not a conversion sign or reference error. The validated controller/solver profile keeps all 12 source relations within `0.000262 rad`, under the unchanged `0.003 rad` gate, with no range violations, NaNs, sign inversion, contact, or active-rollout qpos writes.

**Physical manipulation remains FAIL.** No bottle was present in the diagnostic, no grasp was attempted, and no claim is made that the corrected profile can carry or place the bottle. That requires a separately authorized physical manipulation test after review of this coupling result.
