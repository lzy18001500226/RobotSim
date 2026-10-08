# Issue #46 X2 OmniPicker Dynamic Recovery

**Result: FAIL - Stage 3 no-contact OPEN HOLD crossed the right driver's source lower limit on the first step.** The pinned zero velocity/effort values were correctly reclassified as unspecified simulation metadata. This failure came from the candidate controller's first-step dynamics, not from interpreting those zeros as movement limits.

## Pinned source audit

- X2 model: [`AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`](https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d), `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- OP1 SDK interface: [`AgibotTech/agillink_omnihand_sdk`, commit `c23801f03e396d8d667a307548a05b5d0aecf9aa`](https://github.com/AgibotTech/agillink_omnihand_sdk/blob/c23801f03e396d8d667a307548a05b5d0aecf9aa/linux/x64/cpp/include/omnihand/omnipicker_2025.h), header SHA256 `7298b4f96d179ec62e99876b05641d9ac04b2788972ea626c1c81bc3ab890239`.
- X1 inference actuator interface: [`AgibotTech/agibot_x1_infer`, commit `9e0b818804d644fb9c9663e932dd33b03b24dfa4`](https://github.com/AgibotTech/agibot_x1_infer/blob/9e0b818804d644fb9c9663e932dd33b03b24dfa4/src/module/dcu_driver_module/xyber_controller/xyber_api/include/internal/omni_picker.h), header SHA256 `22f418b2e0640c291b679c72ef0f96708d1e46cc2e866c3993ed4fd94edc25a6`.
- OP1 documents a 1-DOF interface and raw position, velocity, force, acceleration, and deceleration command fields, plus position/velocity/force feedback. Its `SetPositionRatio` comment defines `0` as closed and `1` as open, although this base implementation throws `unsupported`; copied comments elsewhere say 12 motors while the API constant returns 1.
- The X1 API treats OmniPicker as actively commanded: it normalizes position/current commands and exposes velocity feedback. This is corroborating actuator evidence, not an X2 hardware rating.
- Neither interface establishes the exact X2 product variant or its velocity and force calibration. The X2 URDF's four picker fields remain recorded as `velocity=0`, `effort=0` with interpretation `UNSPECIFIED_DYNAMIC_METADATA` for this simulation. The URDF itself was not modified.

## Model and command

The right source driver is `right_claw_joint`, range `[-1, 0]` rad. Its source follower is `R_hand_wide1_joint`, range `[0, 1]` rad, with `q_follower = -q_driver`. MuJoCo 3.3.6 retains the position ranges but does not import the URDF mimic annotation, so the scratch model adds the source joint equality. It exposes one aperture command: `0` closed, `1` open, mapped as `q_driver = -aperture_ratio`; the follower receives no actuator or command.

The unused left pair was initialized at driver `-0.5`, follower `+0.5`, and held with a native constant-joint constraint under `UNUSED_LEFT_PICKER_LOCKED_FOR_RIGHT_HAND_ONLY_M0`. There was no left actuator, left contact, object weld, mocap, object-state write, or active-rollout follower `qpos` write. For picker isolation, the other robot joints were held at the accepted static pose with native equality constraints; this is a simulation fixture, not a claim about free hardware or posture control.

## One derived simulation profile

The compiled full mass matrix at the open pose gives an ideal one-coordinate inertia projection `v^T M v = 0.000253485258061645 kg m^2`, for source tangent `v=[1,-1]`. The corresponding bias projection was `-0.036449145091551 Nm`. This projection assumes exact dependent motion and stationary other coordinates; it does not condense the compliance of the compiled equality constraints. The failed first step shows that it was not an adequate effective inertia for this constraint-compliant model. At a `0.002 s` timestep, one 2.5 s minimum-jerk full-stroke reference implies peak commanded speed `0.75 rad/s` and acceleration `0.92376043 rad/s^2`.

The one attempted profile was critically damped (`zeta=1`) with natural frequency `10 rad/s`, equivalent open-pose gains `kp=0.0253485 Nm/rad`, `kv=0.00506971 Nm s/rad`, and a simulation effort cap of `0.08 Nm`. These are `SIMULATION_ONLY_M0` engineering values, not source or hardware ratings. The source position ranges were unchanged.

## Stage 3 result

The run stopped after its first `mj_step`, at `t=0.002 s`, during `OPEN_HOLD`:

| Measurement | Result |
|---|---:|
| Right driver target / source lower limit | `-1.0 / -1.0 rad` |
| Right driver after step | `-1.0001543257750263 rad` |
| Source-limit overshoot | `0.0001543257750263 rad` |
| Driver velocity / acceleration | `-0.0771629 rad/s / -38.5814 rad/s^2` |
| Follower after step | `0.9999484225 rad`, inside `[0,1]` |
| Mimic residual after step | `-0.0002059033 rad` |
| Applied torque / configured cap | `-0.0364491 / 0.08 Nm` |
| Contact / penetration / nonfinite state | `0 / 0 m / none` |
| Active-rollout follower or object `qpos` writes | `0 / 0` |

Stage 3 failed on the source position limit before CLOSE. The source velocity/effort zeros were not used as bounds. The trace and result also record the acceleration safety bound breach, but the first failed gate is the position-limit crossing. Stages 4-6 (fixed-object contact, bottle hold, and lifts) were not run.

No MP4 was created because no contact, bottle-hold, or lift stage was reached.

## One bounded root-cause diagnostic

A no-step `mj_forward` comparison on the same reset state made zero `mj_step` calls. With zero motor torque, driver/follower accelerations were `+140.8931/-139.9203 rad/s^2`. Applying the computed bias feedforward `-0.0364491 Nm` changed them to `-38.5814/-12.8944 rad/s^2`; applying the opposite positive torque changed them to `+320.3677/-266.9462 rad/s^2`. The actuator's sign/gear path is therefore responsive in the expected direction, but the projected `v^T M v` and `v^T qfrc_bias` feedforward does not produce a valid static hold in the compiled, constraint-compliant system. The driver/follower acceleration pair is no longer kinematically opposite under the candidate torque, indicating that the native equality and other soft pose constraints materially affect the first-step response.

Primary failure classification: **candidate computed-torque model does not match the compliant constrained system at the exact open hard stop**. No sign, gear, or source metadata error was found. Do not use this profile for contact testing. The next design review should decide how to preserve the public endpoint semantics while preventing a one-step hard-limit excursion and how to derive control authority from the actual constrained response; this run does not authorize a controller retune.

## Reproduction and evidence

The recovery runner is [`scripts/research/issue46_omnipicker_recovery.py`](../../scripts/research/issue46_omnipicker_recovery.py). Its exact command is preserved in the external evidence packet. Raw `result.json`, `no_contact_trace.jsonl`, source/header audit, controller derivation, run log, and accepted static PNGs are under `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-dynamic\20261007-zero-metadata-recovery-v1\`.

The X2 OmniPicker position topology and mimic relation are source-derived. The velocity/effort bounds used here are SIMULATION_ONLY_M0 engineering parameters because the pinned X2 URDF provides zero placeholders and an exact X2 hardware dynamic calibration has not been established.
