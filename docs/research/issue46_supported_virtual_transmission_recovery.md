# Issue #46 Supported Virtual-Transmission Recovery

## Final endpoint-margin iteration: FAIL at representative thumb settling

**FAIL - Stage 2 representative case 2 did not settle to the operational OPEN hold.** This was the final authorized OmniHand repair iteration. The follower-aware endpoint margin prevented the earlier source hard-limit crossing, but the thumb follower remained above the fixed final-settling speed criterion. The run stopped at that first failed case. Full-hand prevalidation, bottle HOLD, and the 1 mm / 5 mm extension were not run.

### Endpoint margin derivation

The prior smooth-return evidence showed an `R_thumb_dip_joint` lower-limit crossing of `8.62906053e-6 rad`, maximum target tracking error `0.00285180638 rad`, maximum mimic residual `0.00137822456 rad`, terminal reference speed `4.37484653e-7 rad/s`, peak hand speed `0.0296628853 rad/s`, and peak acceleration `0.113792258 rad/s^2` at `dt=0.002 s`. The one-step integration allowance was `5.95533551e-5 rad`.

The single derived margin was:

```text
max(0.00198424251 rad endpoint target/state excursion,
    0.00285180638 rad maximum tracking error)
+ 0.00137822456 rad maximum mimic residual
+ 0.000000000875 rad terminal target travel
+ 0.00005955336 rad one-step integration allowance
= 0.00428958517 rad, rounded up to 0.00430 rad
```

The former operational margin was `0.00150 rad`; the new operational margin is `0.00430 rad`. Source hard joint limits were unchanged. The margin was applied to active target intervals and derived follower target intervals. All 32 saved kp/kv entries matched the frozen controller profile exactly; the controller design margin remained `0.00150 rad`, and the wn*dt, damping, effort bounds, and mimic mapping were unchanged. Of the 32 hand joints, 22 were near an OPEN endpoint. The margin is 0.503% of the smallest source range among those endpoint-near joints (`0.85521133 rad`); the narrowest source range across all hand joints is `0.17453293 rad`.

Examples from the endpoint audit: an active finger PIP OPEN target moved from `0.00150` to `0.00430 rad`; its 1.097-ratio DIP follower target moved from `0.0016455` to `0.0047171 rad`. These are operational targets, not altered source limits.

### Representative cases and first failure

- **Case 1, `R_index_dip` follower: PASS.** The `R_index_pip_joint` driver moved `+0.01 rad`, returned smoothly, and settled. Maximum mimic residual was `0.00083163471 rad`, maximum target error `0.00144888958 rad`, peak hand speed `0.0283854421 rad/s`, peak acceleration `0.113788752 rad/s^2`, and maximum effort `0.03953487 Nm`. There were no limit violations, self-penetration, contacts, or follower qpos writes.
- **Case 2, thumb follower: FAIL.** The `R_thumb_mcp_joint` driver moved `+0.01 rad`; the 1.097-ratio `R_thumb_pip_joint` follower remained in motion after the 250-step final OPEN hold. The fixed gate allows final-25-step maximum speed no greater than 10% of peak hand speed: `0.00296615870 rad/s`. The measured final-25-step maximum was `0.00650454394 rad/s` (2.1929x the limit). Final target error was `0.00053244303 rad`, under its `0.010 rad` bound. Whole-case maximum mimic residual was `0.00137855672 rad`, peak hand speed `0.0296615870 rad/s`, peak acceleration `0.1058976162 rad/s^2`, and maximum effort `0.03953487 Nm`, under the unchanged `0.05 Nm` simulation bound.
- The failed thumb case had zero hard position-limit violations, zero velocity-limit violations, zero self-penetration, zero contacts, finite state, no effort or mimic-abort violations, and zero active-rollout follower qpos writes. At the final sample, hand speed was `0.00545332116 rad/s`, acceleration `0.0246866792 rad/s^2`, mimic residual `0.00028080609 rad`, and target error `0.00053244303 rad`.
- The permitted bounded diagnosis read the already-saved trace only. It confirmed residual thumb follower motion exceeded the fixed settling criterion; no additional simulation or parameter change was made. The remaining two representative cases did not run.

### Stop and runtime identity

This result is a settling-time failure after the endpoint correction, not a source-limit crossing or bottle-contact result. Per the final-iteration stop rule, Stage 3 full-hand prevalidation, Stage 4 bottle HOLD, and the 1 mm / 5 mm extension were not run. No bottle contact was attempted. This does not establish a bottle failure or justify an architecture switch; the OmniHand path remains uncleared for manipulation pending maintainer review.

- RobotSim base HEAD: `e95749715a58405c4c352ece565b6d44c8808de1`.
- AgiBot vendor pin and observed HEAD: `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- Python 3.10.12; MuJoCo Python/native 3.3.6; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`; WSL2 Linux x86_64; timestep `0.002 s`.

### Reproduction and evidence

The final iteration was invoked with:

```bash
set -o pipefail
cd /tmp/robotsim-issue46-virtual-transmission-resume-20261007
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  ISSUE46_PREVIOUS_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/smooth-reference-recovery-03 \
  ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/final-endpoint-margin-04 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u \
  scripts/research/issue46_virtual_transmission_supported_recovery.py --final-endpoint-margin \
  2>&1 | tee /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/final-endpoint-margin-04/run.log
```

The output directory did not exist when `tee` opened `run.log`; the harness created it later, so `tee` reported that it could not open the log. The authorized simulation completed and wrote its JSON/JSONL evidence, but **no `run.log` exists**. This is a logging gap; the console summary was retained in the task output. Do not treat a reconstructed log as raw evidence.

The external packet is `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/final-endpoint-margin-04/`. It contains `final_margin_derivation.json`, `endpoint_target_audit.json`, the frozen controller profile and comparison, `stage3_representative_hand_result.json`, the 140 MB `stage3_representative_hand_trace.jsonl`, `failure_diagnosis.json`, `final_iteration_result.json`, and `runtime_identity.json`. The raw files were not edited after collection. PR #53 remains Draft; no merge or bottle experiment occurred.

## Latest iteration: smooth-reference recovery

**FAIL - Stage 3 source hard position limit.** The 1.0 s minimum-jerk reference passed the representative `R_index_dip` follower case, resolving the previously observed 0.000106315 rad index-follower lower-limit crossing. The second case, a `+0.01 rad` command to `R_thumb_mcp_joint`, crossed the unchanged lower limit of `R_thumb_dip_joint` during return to OPEN at step 1170 / 2.340 s. The measured position was `-0.00000862906 rad` against a 0 rad lower limit. The run stopped there; the remaining representative cases, full-hand prevalidation, and bottle contact were not run.

### Gates and first failure

- **Stage 1 audit: PASS.** The original 422-step index return trace was replayed with detailed torque/state instrumentation and matched the saved trace exactly (`max_abs_follower_position_delta_rad=0`). It classified the original index failure as insufficient dynamic margin at the operational OPEN endpoint: the derived `R_index_dip_joint` target was 0.0016455 rad above its 0 rad hard lower limit, while its measured return undershoot was 0.00175181548 rad, producing the recorded 0.00010631548 rad crossing at step 422 / 0.844 s. Targets remained inside operational soft limits.
- **Stage 2 supported OPEN hold: PASS, reused.** The previously accepted 1,000-step / 2.0 s hold evidence was copied unchanged into this packet; the stage was not rerun.
- **Reference choice.** One 1.0 s quintic minimum-jerk transition was selected without a duration sweep. For a 0.01 rad driver move its peak target speed is 0.01875 rad/s and peak acceleration is about 0.057735 rad/s^2; the 1.097 follower ratio gives 0.0205685 rad/s and 0.0633341 rad/s^2. Those targets are far below the relevant 21 rad/s source velocity limit. The prior 0.2 s cubic return reached 0.082264 rad/s at the follower while undershooting the 0.0016455 rad OPEN margin, so the slower profile was selected using that measured response and endpoint margin. The passing index case then measured peak actual speed 0.0283864 rad/s and decayed to 0.00201690 rad/s over its final 25 steps.
- **Stage 3 case 1: PASS.** `R_index_pip_joint` drove the `R_index_dip_joint` follower by +0.01 rad, followed by a 1.0 s minimum-jerk return and hold. Peak mimic residual was 0.000831738 rad, peak hand speed 0.0283864 rad/s, peak acceleration 0.113792 rad/s^2, maximum effort 0.0395442 Nm, and maximum target error 0.00144909 rad. It had no position/velocity-limit violations, contacts, or follower qpos writes; the final 25-step response was damped.
- **Stage 3 case 2: FAIL.** The `R_thumb_mcp_joint` +0.01 rad case failed on `R_thumb_dip_joint` at step 1170 / 2.339999999999964 s in `source_return`. At the first failure, the source target was 0.00197561345 rad with target velocity -0.00127963 rad/s; actual position was -0.00000862906 rad, velocity -0.00661930 rad/s, and acceleration 0.0882758 rad/s^2. The source lower limit was 0 rad, so the crossing was 0.00000862906 rad. The target was inside its operational range. The failure snapshot had mimic residual 0.000635097 rad and target tracking error 0.00198424 rad; the whole trial maximum mimic residual was 0.00137822 rad, below the 0.010 rad manipulation abort ceiling. There were no velocity-limit violations, contacts, initial penetration, non-finite values, or active-rollout follower qpos writes. Peak trial hand speed/acceleration were 0.0296629 rad/s and 0.105908 rad/s^2; maximum effort was 0.0395442 Nm, within the existing 0.05 Nm bound.

This failure is a first-gate stop, not evidence that the remaining reference cases or contact behavior pass. No gains, effort limits, source limits, mimic mapping, or physical model were changed. No bottle contact occurred. No further diagnosis or tuning was run after the first-failure state snapshot.

### Runtime provenance

- Python 3.10.12; MuJoCo Python/native 3.3.6; timestep 0.002 s; WSL2 Linux x86_64.
- Native MuJoCo library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- AgiBot vendor pin and observed HEAD: `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- The carried-forward `runtime_identity.json` records RobotSim base HEAD `2347e76beab320f8da3089514ff4086201f5ee67` from the accepted Stage 1/2 packet. The Stage 3 continuation itself ran from task checkout HEAD `3469d335632eb40d8e53bd65ec4c2828ef6ff541` with the research harness modified for this recovery. The identity file was preserved unchanged and does not identify that harness diff.

### Reproduction commands

Stage 1 saved-trace audit:

```bash
cd /tmp/robotsim-issue46-virtual-transmission-resume-20261007
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  ISSUE46_PREVIOUS_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/supported-fixed-body-recovery-02 \
  ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/smooth-reference-recovery-03 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u \
  scripts/research/issue46_virtual_transmission_supported_recovery.py --audit-original-stage3
```

Stage 3 continuation using the preserved Stage 1/2 evidence:

```bash
cd /tmp/robotsim-issue46-virtual-transmission-resume-20261007
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/smooth-reference-recovery-03 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u \
  scripts/research/issue46_virtual_transmission_supported_recovery.py --continue-after-stage2
```

### Evidence packet

The external evidence directory is:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/smooth-reference-recovery-03/`

It contains the preserved Stage 1/2 JSON and traces, `stage1_return_overshoot_audit.json` and its full JSONL trace, `stage3_representative_hand_result.json` and its full JSONL trace, `supported_recovery_result.json`, the runtime identity, and the complete Stage 3 log. The raw evidence was not edited after collection. PR #53 remains Draft; no merge or bottle experiment was performed.

## Previous unsmoothed Stage 3 result

**FAIL - Stage 3 source hard position limit.** The supported fixed-body setup passed its initial audit and 2 s operational OPEN hold. The first representative hand command then drove `R_index_dip_joint` 0.000106315 rad below its unchanged source lower limit while returning to OPEN. The task stopped at that first failed gate. No controller tuning, bottle contact, or later-stage experiment was performed.

### Runtime identity

- RobotSim base HEAD observed by the run: `2347e76beab320f8da3089514ff4086201f5ee67`; the tested working tree also contained the research harness changes included with this report.
- AgiBot source: `575cc6b988f976c23550e0db85aa1e5475d3652d`, matching the pinned vendor checkout.
- Python 3.10.12; MuJoCo Python/native 3.3.6; timestep 0.002 s; WSL2 Linux x86_64.
- Native library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.

### Gate results

#### Stage 1 - support audit: PASS

The bottle-disabled model had `nq=nv=nu=63`, `neq=0`. The base was fixed by model topology. All 31 non-hand robot DOFs were classified and supported: 12 leg, 3 waist, 2 head, and 14 arm joints. Existing source-bounded position actuators held the reference posture; no support gain or source-limit changes were made. Gravity compensation shifted the control target by `qfrc_bias / (gear^2 * kp)` and clamped the result to each source hard range. Support target groups used the existing 250/30 (non-arm) and 150/22 (arm) position/velocity gains; actuator effort ranges remained source-bounded. There were no intentionally free robot DOFs, no initial self-penetration, and no rollout qpos writes.

#### Stage 2 - supported OPEN hold: PASS

The clean supported hold completed 1,000 steps (2.0000000000000013 s). It had zero source position or velocity violations, finite state, no self-penetration, and zero post-start qpos writes. Maximum support effort was 2.1770209723 Nm; maximum hand effort was 0.0395441485 Nm against the existing 0.05 Nm simulation bound. Maximum hand speed/acceleration were `3.01e-15 rad/s` and `1.77e-12 rad/s^2`. Maximum mimic residual was `5.33e-16 rad`. Maximum supported motion was 0.000284444 rad in the legs and below `1.24e-18 rad` in the other groups.

#### Stage 3 - representative hand cases: FAIL, stopped on first case

Only the first required case ran: a source-valid `+0.01 rad` step to `R_index_pip_joint`, then return and OPEN hold. At step 422 / 0.844 s during `source_return`, `R_index_dip_joint` had `q=-0.0001063154808 rad` against source limits `[0, 1.8325957146] rad`; velocity was `-0.11754156 rad/s`, within its 21 rad/s limit. The lower-limit overshoot was 0.000106315 rad.

There were no contacts, self-contacts, velocity-limit violations, NaNs, or active-rollout follower qpos writes. Maximum hand effort was 0.03954416 Nm; maximum support effort was 2.17709660 Nm. Maximum hand qvel/qacc were 0.148257 rad/s and 1.252878 rad/s^2. Maximum mimic residual was 0.00570335 rad: above the 0.003 rad diagnostic target but below the 0.010 rad engineering abort ceiling. Maximum target tracking error was 0.00970213 rad.

One read-only decomposition at the first failing state found `qfrc_actuator=-9.23185e-5 Nm`, `qfrc_bias=-9.52740e-5 Nm`, `qfrc_constraint=0`, hand mass diagonal `1.51156e-6 kg m^2`, and maximum whole-body coupling into the hand of `3.12481e-8 Nm`. This localizes the observed failure to a return/open transient at the source hard stop; it is not a contact or unsupported-body-collapse failure. No corrective change was applied.

The remaining three representative cases, Stage 4 full-hand prevalidation, and Stage 5 bottle contact were **NOT RUN**. Bottle geometry/contact was absent from this diagnostic model. Do not interpret this result as a virtual-transmission manipulation pass.

### Reproduction

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

### Evidence

Raw logs, JSON results, and full JSONL traces are preserved outside the repository at:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/supported-fixed-body-recovery-02/`

The packet includes runtime identity, Stage 1 support audit, Stage 2 result/trace, Stage 3 result/trace, aggregate result, and both initial/final run logs. PR #53 remains Draft; no merge was performed.
