# Issue #46 Virtual-Transmission Soft-Limit Recovery

## Result

**FAIL — source hard-limit violation during OPEN hold at 0.002 s after applying the operational target margin.** Stage 1 classified the prior failure as target-at-hard-stop transient (A). The target-only margin passed static validation but did not prevent the actual state from moving outward on the first step. One bounded force-balance diagnosis was run. No bottle-contact or lift stage ran. PR #53 remains Draft.

## Runtime Identity

- Worktree HEAD: `2cd3befd3b6ca508ea9363d3fdbaef65ddf19235`
- MuJoCo: `3.3.6`
- Timestep: `0.002 s`
- X2 source: `575cc6b988f976c23550e0db85aa1e5475d3652d`
- Model label: `CONTROL-LEVEL / ACTUATED MIMIC EMULATION`

## Stage 1: Hard-Limit / Target Audit

The preserved failed run and a clean one-step reproduction both show all 12 OPEN targets equal to a source hard-limit endpoint. At reset, source and raw qpos were zero, qvel was zero, mimic residual was zero, and there was no self-contact or penetration. First-step actuator efforts were zero because position and velocity errors were zero at the reset force evaluation. The initial dynamics then moved these coordinates outward.

| Joint | Source hard range (rad) | OPEN target (rad) | Initial qpos source/raw | First qpos source/raw | First qvel (rad/s) | Actuator effort (N m) | Target endpoint | Overshoot (rad) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| L_index_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000364831 / -0.000364831 | -0.182415484 | 0 | lower | 0.000364831 |
| L_middle_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000368386 / -0.000368386 | -0.184193083 | 0 | lower | 0.000368386 |
| L_pinky_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000370882 / -0.000370882 | -0.185440866 | 0 | lower | 0.000370882 |
| L_ring_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000369600 / -0.000369600 | -0.184800091 | 0 | lower | 0.000369600 |
| L_thumb_mcp_joint | [-0.855211333, 0] | 0 | 0 / 0 | 0.000289070 / 0.000289070 | 0.144534940 | 0 | upper | 0.000289070 |
| L_thumb_pip_joint | [-1.134464014, 0] | 0 | 0 / 0 | 0.000616201 / 0.000616201 | 0.308100574 | 0 | upper | 0.000616201 |
| R_index_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000340080 / -0.000340080 | -0.170040032 | 0 | lower | 0.000340080 |
| R_middle_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000348999 / -0.000348999 | -0.174499892 | 0 | lower | 0.000348999 |
| R_pinky_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000363038 / -0.000363038 | -0.181518854 | 0 | lower | 0.000363038 |
| R_ring_dip_joint | [0, 1.832595715] | 0 | 0 / 0 | -0.000355683 / -0.000355683 | -0.177841572 | 0 | lower | 0.000355683 |
| R_thumb_mcp_joint | [0, 0.855211333] | 0 | 0 / 0 | -0.000057838 / -0.000057838 | -0.028918874 | 0 | lower | 0.000057838 |
| R_thumb_pip_joint | [0, 1.134464014] | 0 | 0 / 0 | -0.000626846 / -0.000626846 | -0.313422769 | 0 | lower | 0.000626846 |

Classification: **A — target commanded exactly at hard stop plus initial dynamic transient.** This is not an out-of-range target, inconsistent initialization, or coupling-induced first excursion. The observed worst crossing was `0.000626846 rad` on `R_thumb_pip_joint`; the maintainer's `0.000289 rad` matches only `L_thumb_mcp_joint`.

## Stage 2: Operational Soft Target

One simulation/control margin was selected: `0.0015 rad` on either side of the source limits. It is `2.39x` the measured maximum one-step crossing, leaving `0.000873 rad` of measured headroom, and is `0.86%` of the narrowest hand joint range (`0.174533 rad`). This changes target validation only. Source limits, compiled joint ranges, and qpos are unchanged.

The public active-joint API and derived follower targets both enforce the operational interval. OPEN, PARTIAL, and CLOSE target-only preflight passed. It changed no qpos and had no initial contacts, penetration, hard-limit violation, or mimic residual.

## Stage 3: Full Prevalidation

The clean-reset OPEN hold failed at step 1 (`0.002 s`). Twelve coordinates crossed source hard limits despite active and follower targets being inside the operational interval. Maximum outward excursion was `0.001123435 rad` on `R_thumb_pip_joint`. There were no contacts, NaNs, velocity-limit violations, or initial self-penetrations. Maximum mimic residual was `0.004699200 rad`, below the `0.010 rad` manipulation abort ceiling but above the `0.003 rad` high-fidelity diagnostic target.

The single bounded diagnosis recorded applied actuator forces and generalized bias forces. Active actuator gear and target signs were correct. The `0.0015 rad` target inset generated only `0.00003 N m` from the active position servos. At reset, bias torque was approximately `0.0006 N m` on the finger PIPs and `0.0207 N m` on the left thumb MCP. The controller did not have enough initial authority to move inward against that load; qpos crossed the hard range before the controller could recover. Follower motors likewise remained under their `0.05 N m` force bound, but their initial response did not prevent the follower crossings.

This failure is **not** evidence that the virtual-transmission architecture is rejected. It does establish that the tested target margin alone is insufficient with the current simulation-derived servo gains and reset pose. No further gain or margin tuning was done.

## Stages Not Run

The one-failure stop rule was reached during Stage 3. Bottle contact hold, 30 mm lift, 3-run repeatability, and ±5 mm target perturbation were not run. No bottle qpos writes, attachment, or contact occurred.

## Evidence And Reproduction

Evidence directory:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/soft-limit-audit/`

- `hard_limit_target_audit.json`: exact 12-joint Stage 1 values and actuator efforts.
- `soft_limit_static_preflight.json`: OPEN/PARTIAL/CLOSE target interval checks.
- `soft_limit_failure_diagnosis.json`: one-step actuator, bias, and qacc evidence.
- `../prevalidation-soft-limits-01/virtual_transmission_prevalidation_result.json`: Stage 3 result.
- `../prevalidation-soft-limits-01/virtual_transmission_prevalidation_trace.jsonl`: first failed step trace.
- `../prevalidation-soft-limits-01.run.log`: complete Stage 3 console log.

Stage 3 reproduction command:

```bash
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/prevalidation-soft-limits-01 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u \
  /tmp/robotsim-issue46-virtual-transmission-resume-20261007/scripts/research/issue46_virtual_transmission_gate.py
```
