# Issue #46 X2 OmniPicker Endpoint Recovery

**Result: FAIL at Stage 5, the first step of the two-second no-contact operational OPEN hold.** The driver-only constrained controller cancels driver acceleration at reset, but the source follower retains `-40.1820 rad/s^2` acceleration through the active, compliant native mimic equality. This exceeds the predeclared SIMULATION_ONLY_M0 acceleration bound; no later dynamics or contact stage was run.

This is a separate attempt from the prior endpoint failure in [`issue46_omnipicker_dynamic_recovery.md`](issue46_omnipicker_dynamic_recovery.md). It preserves the source hard limits, native mimic equality, one active driver, zero follower actuators, and zero active-rollout follower `qpos` writes.

## Stage 1: Initial State Audit

At the prior exact OPEN initialization, `right_claw_joint=-1.0 rad` and `R_hand_wide1_joint=+1.0 rad`, both inside their source ranges. The equality was active and the mimic residual was zero; both velocities were zero. With the reset actuator command at zero, `mj_forward` predicted driver/follower accelerations of `+140.8931/-139.9203 rad/s^2`. The driver/follower bias forces were `-0.0113854/+0.0250637 Nm`, passive forces were zero, actuator forces were zero, and the equality row force was `-0.000398228 Nm`.

The previous controller's first command was its projected bias feedforward, `-0.0364491 Nm`. At the unchanged reset state, this changed driver/follower accelerations to `-38.5814/-12.8944 rad/s^2`, with equality force `+0.0210721 Nm`. Thus the original initialization was source-valid and kinematically consistent but not dynamically supported; the old feedforward accelerated the driver outward from the lower hard stop. The raw runner's Stage 1 `PASS` checks only source validity, equality consistency, and zero velocity, so it does not mean the full requested dynamic-support condition passed. That unsupported response was retained and tested again after deriving the operational endpoint and equilibrium command.

## Stage 2: Operational OPEN Margin

The one margin used the previous excursion and equality residual:

```text
observed envelope = 0.000154325775 + 0.000205903302 = 0.000360229077 rad
2x envelope       = 0.000720458154 rad
round upward to the next 0.001 rad quantum = 0.001 rad
```

The source hard OPEN limit remains `-1.0 rad`; the operational target is `-0.999 rad`, with follower target `+0.999 rad` derived from the source mimic relation. The margin is SIMULATION_ONLY_M0 control metadata, not a changed hardware limit. The accepted OPEN jaw link-origin separation changed from `124.448933 mm` to `124.400705 mm`, a `0.048229 mm` reduction; this is a link-origin measure, not surface aperture.

The public aperture still has one scalar command: `0` is CLOSED and `1` is operational OPEN. The mapped driver target is `-0.999 * aperture_ratio`; the follower remains equality-derived.

## Stages 3-4: Local Model And Single Controller

The local response used fresh no-step `mj_forward` evaluations at the operational OPEN state, with the native equality and accepted fixed-pose equalities active. A bounded `+/-0.004 Nm` torque difference gave:

| Quantity | Result |
|---|---:|
| Driver acceleration at zero torque | `+140.812237 rad/s^2` |
| Follower acceleration at zero torque | `-139.844311 rad/s^2` |
| Driver `d(qacc)/d(torque)` | `4923.973445 rad/s^2/Nm` |
| Follower `d(qacc)/d(torque)` | `-3485.027110 rad/s^2/Nm` |
| Local driver effective constrained inertia | `0.000203088016 kg m^2` |
| Torque that cancels driver acceleration | `-0.028597278 Nm` |
| Follower acceleration at that torque | `-40.182023 rad/s^2` |

One critically damped profile was derived at `wn=5 rad/s`: `kp=0.00507720 Nm/rad`, `kd=0.00203088 Nm s/rad`, with the existing `0.08 Nm` SIMULATION_ONLY_M0 effort cap. The 2.5 s minimum-jerk, `0.999 rad` reference has predicted peak target velocity `0.74925 rad/s` and acceleration `0.922837 rad/s^2`; the corresponding predeclared M0 state bounds were `1.123875 rad/s` and `1.845673 rad/s^2`. These are simulation engineering bounds, not AgiBot ratings.

The local finite-difference model is not an exact reduced-coordinate model. It cancels the driver's acceleration only; it cannot simultaneously cancel the unactuated follower's acceleration because the compiled source equality is compliant.

## Stage 5: OPEN HOLD Result

The run initialized directly at `(-0.999, +0.999) rad`, zero velocity, equality active, and the derived equilibrium command already set before the first `mj_step`. At step 1 (`2 ms`):

- driver acceleration was approximately `0 rad/s^2`;
- follower acceleration was `-40.182023 rad/s^2`, versus the `1.845673 rad/s^2` M0 bound;
- follower velocity reached `-0.0803640 rad/s` and its position became `0.998839272 rad`;
- mimic residual was `-0.000160728 rad`;
- applied driver torque was `-0.0285973 Nm`, inside the `0.08 Nm` cap;
- there were no contacts, hard position-limit violations, NaNs, object writes, or active-rollout follower `qpos` writes.

The single bounded post-failure diagnosis made zero additional `mj_step` calls. At the same failure state, zero torque produced driver/follower accelerations `+148.284661/-141.582238 rad/s^2`; the last controller command produced `+8.168414/-41.571491 rad/s^2`. The equality force changed from `0.000909767 Nm` to `0.017811064 Nm`. This confirms the controller reduces driver acceleration but leaves a large follower response under the compliant relation.

Primary classification: **driver-only control plus compliant source equality does not provide acceleration-level follower support under gravity**. This is a real no-contact OPEN-hold failure, not a grasp/contact failure. No parameter sweep, second controller family, CLOSE, fixed-object contact, bottle hold, or lift was attempted.

## Stage Status And Evidence

| Stage | Status |
|---|---|
| Initial state consistency audit | Source position, equality residual, and zero velocity PASS; dynamic support FAIL at zero command and remained incomplete under the later driver-only equilibrium command |
| Single operational OPEN margin | Derived: `0.001 rad` |
| Constrained local dynamics | Measured with no-step finite differences |
| One controller profile | Derived; not accepted as stable |
| Two-second OPEN hold | **FAIL at step 1** |
| No-contact CLOSE/REOPEN | NOT RUN |
| Fixed-object contact | NOT RUN |
| Bottle hold and 1/5/30 mm lifts | NOT RUN |

Raw evidence is preserved externally at `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-dynamic\20261007-constrained-open-hold-v3\`, including `initial_state_audit.json`, `operational_open_margin.json`, `constrained_local_dynamics.json`, `controller_profile.json`, `open_hold_trace.jsonl`, `open_hold_failure_diagnostic.json`, `result.json`, `run.log`, the exact command, pinned-source audit, and accepted static images. No MP4 was produced because the run stopped before contact.

Exact reproduction command (use a new empty output directory):

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_omnipicker_endpoint_recovery.py --source-root /tmp/robotsim-issue46-virtual-transmission-vendor-20261007 --sdk-root /tmp/robotsim-issue46-agillink-c238 --x1-infer-root /tmp/robotsim-issue46-x1-infer-9e0 --output <new-empty-output-directory>
```

Source position topology and mimic relation are source-derived. The velocity/effort bounds used here are `SIMULATION_ONLY_M0` engineering parameters because the pinned X2 URDF provides zero placeholders and an exact X2 hardware dynamic calibration has not been established.

Final result: **FAIL — Stage 5 OPEN HOLD exceeded the derived picker acceleration bound on the first step; follower acceleration was `-40.182023 rad/s^2` at 2 ms.**

Temporary research runner disposition: `KEEP-AS-TOOL` for bounded, offline constrained-response diagnostics. It is not production control code and is not imported by robot runtime.
