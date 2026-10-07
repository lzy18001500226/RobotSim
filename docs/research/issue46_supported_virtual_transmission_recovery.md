# Issue #46 Supported Virtual-Transmission Recovery

## Result

**FAIL - Stage 3 source hard position limit.** The supported fixed-body setup passed its initial audit and 2 s operational OPEN hold. The first representative hand command then drove `R_index_dip_joint` 0.000106315 rad below its unchanged source lower limit while returning to OPEN. The task stopped at that first failed gate. No controller tuning, bottle contact, or later-stage experiment was performed.

## Runtime identity

- RobotSim base HEAD observed by the run: `2347e76beab320f8da3089514ff4086201f5ee67`; the tested working tree also contained the research harness changes included with this report.
- AgiBot source: `575cc6b988f976c23550e0db85aa1e5475d3652d`, matching the pinned vendor checkout.
- Python 3.10.12; MuJoCo Python/native 3.3.6; timestep 0.002 s; WSL2 Linux x86_64.
- Native library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.

## Gate results

### Stage 1 - support audit: PASS

The bottle-disabled model had `nq=nv=nu=63`, `neq=0`. The base was fixed by model topology. All 31 non-hand robot DOFs were classified and supported: 12 leg, 3 waist, 2 head, and 14 arm joints. Existing source-bounded position actuators held the reference posture; no support gain or source-limit changes were made. Gravity compensation shifted the control target by `qfrc_bias / (gear^2 * kp)` and clamped the result to each source hard range. Support target groups used the existing 250/30 (non-arm) and 150/22 (arm) position/velocity gains; actuator effort ranges remained source-bounded. There were no intentionally free robot DOFs, no initial self-penetration, and no rollout qpos writes.

### Stage 2 - supported OPEN hold: PASS

The clean supported hold completed 1,000 steps (2.0000000000000013 s). It had zero source position or velocity violations, finite state, no self-penetration, and zero post-start qpos writes. Maximum support effort was 2.1770209723 Nm; maximum hand effort was 0.0395441485 Nm against the existing 0.05 Nm simulation bound. Maximum hand speed/acceleration were `3.01e-15 rad/s` and `1.77e-12 rad/s^2`. Maximum mimic residual was `5.33e-16 rad`. Maximum supported motion was 0.000284444 rad in the legs and below `1.24e-18 rad` in the other groups.

### Stage 3 - representative hand cases: FAIL, stopped on first case

Only the first required case ran: a source-valid `+0.01 rad` step to `R_index_pip_joint`, then return and OPEN hold. At step 422 / 0.844 s during final OPEN hold, `R_index_dip_joint` had `q=-0.0001063154808 rad` against source limits `[0, 1.8325957146] rad`; velocity was `-0.11754156 rad/s`, within its 21 rad/s limit. The lower-limit overshoot was 0.000106315 rad.

There were no contacts, self-contacts, velocity-limit violations, NaNs, or active-rollout follower qpos writes. Maximum hand effort was 0.03954416 Nm; maximum support effort was 2.17709660 Nm. Maximum hand qvel/qacc were 0.148257 rad/s and 1.252878 rad/s^2. Maximum mimic residual was 0.00570335 rad: above the 0.003 rad diagnostic target but below the 0.010 rad engineering abort ceiling. Maximum target tracking error was 0.00970213 rad.

One read-only decomposition at the first failing state found `qfrc_actuator=-9.23185e-5 Nm`, `qfrc_bias=-9.52740e-5 Nm`, `qfrc_constraint=0`, hand mass diagonal `1.51156e-6 kg m^2`, and maximum whole-body coupling into the hand of `3.12481e-8 Nm`. This localizes the observed failure to a return/open transient at the source hard stop; it is not a contact or unsupported-body-collapse failure. No corrective change was applied.

The remaining three representative cases, Stage 4 full-hand prevalidation, and Stage 5 bottle contact were **NOT RUN**. Bottle geometry/contact was absent from this diagnostic model. Do not interpret this result as a virtual-transmission manipulation pass.

## Reproduction

From a clean evidence directory, run the current research harness with the pinned vendor/runtime paths:

```bash
cd /tmp/robotsim-issue46-virtual-transmission-resume-20261007
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/supported-fixed-body-recovery-02 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u \
  scripts/research/issue46_virtual_transmission_supported_recovery.py
```

The recorded Stage 3 continuation reused the passing Stage 1/2 JSON and traces by adding `--continue-after-stage2` to that command. Do not rerun it to reproduce this report unless a new experiment is authorized.

Exact Stage 3-only continuation command used with the preserved passing Stage 1/2 evidence:

```bash
cd /tmp/robotsim-issue46-virtual-transmission-resume-20261007
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/supported-fixed-body-recovery-02 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u \
  scripts/research/issue46_virtual_transmission_supported_recovery.py --continue-after-stage2
```

## Evidence

Raw logs, JSON results, and full JSONL traces are preserved outside the repository at:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/supported-fixed-body-recovery-02/`

The packet includes runtime identity, Stage 1 support audit, Stage 2 result/trace, Stage 3 result/trace, aggregate result, and both initial/final run logs. PR #53 remains Draft; no merge was performed.
