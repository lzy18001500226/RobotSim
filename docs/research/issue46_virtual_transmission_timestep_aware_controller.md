# Issue #46 Timestep-Aware Virtual-Transmission Controller

## Result

**FAIL — Stage 2 source position-limit gate.** The inertia-scaled controller's first representative case crossed source hard limits during the initial OPEN hold at 6 ms, before the commanded source-coordinate step. The one permitted bounded diagnosis is complete. Stop here: stages 3–7 and all bottle-contact work were not run.

This does not reject the control-level virtual-transmission architecture. The current run instead exposes a non-equilibrated whole-body initialization: unactuated non-hand coordinates accelerate under gravity, and articulated coupling drives hand joints that start only 0.0015 rad inside hard stops.

## Runtime Identity

- RobotSim base HEAD used by the run: `8ba8d241cdc0eff7c3ab59ba063aa22323b119c3`; the task scripts had local modifications recorded by SHA-256 in the evidence packet.
- X2 vendor source: `AgibotTech/agibot_x2_urdf`, pinned commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`, SHA-256 `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- Python `3.10.12`; MuJoCo Python/native `3.3.6`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Timestep `0.002 s` (500 Hz); model label `CONTROL-LEVEL / ACTUATED MIMIC EMULATION`.

## Stage 1: Controller Design

At the clean operational OPEN reset, the full compiled articulated mass matrix `M` was expanded with MuJoCo's mass-matrix data. For each hand DOF `i`, the scalar free-coordinate effective inertia was computed as `I_eff = 1 / (M^-1)[i,i]`. This is the diagonal response to a unit generalized torque at that coordinate with other model coordinates free to respond; it is not a link-only inertia.

All 32 active and follower coordinates used `zeta = 1`, `wn = 25 rad/s` (3.979 Hz), so `wn*dt = 0.05`, or 125.66 samples per natural period. Gains were `kp = I_eff*wn^2` and `kd = 2*zeta*I_eff*wn`. The existing simulation effort bound remained `0.05 N m`; URDF source effort limits remained `33.5 N m`. No source limits, actuator bounds, or operational soft-limit margins were changed.

For the semi-implicit, zero-order-held sampled double-integrator design model, the state matrix was `[[0.9975, 0.0018], [-1.25, 0.9]]`; poles were `0.96` and `0.9375` (maximum magnitude `0.96`). This is an analytical design check, not a substitute for the MuJoCo response gate that failed below.

| Representative DOF | Role | `I_eff` (kg m^2) | `wn` (rad/s) | `kp` (N m/rad) | `kd` (N m s/rad) | source effort / simulation bound (N m) |
|---|---|---:|---:|---:|---:|---:|
| `R_index_dip_joint` | follower | `4.26344e-7` | 25 | `2.66465e-4` | `2.13172e-5` | 33.5 / 0.05 |
| `L_pinky_dip_joint` | follower | `4.26125e-7` | 25 | `2.66328e-4` | `2.13062e-5` | 33.5 / 0.05 |
| `R_index_pip_joint` | active | `3.26600e-6` | 25 | `0.00204125` | `0.000163300` | 33.5 / 0.05 |
| `R_thumb_abad_joint` | active | `1.77235e-4` | 25 | `0.110772` | `0.00886175` | 33.5 / 0.05 |

The evidence `controller_design.json` contains these values for every DOF, including inertia, gains, source effort limit, simulation torque bound, effort-limited bandwidth, discrete poles, and predicted acceleration. For the representative position errors, the nominal model predicts `0.2541 rad/s^2` at `0.00040655 rad`, `0.9375 rad/s^2` at the `0.0015 rad` operational margin, and `6.25 rad/s^2` at `0.01 rad`. At `0.00040655 rad` plus `0.203276 rad/s` velocity error, it predicts `10.4179 rad/s^2`.

## Stage 2: First Representative Case

The required first case, `R_index_dip_joint` follower driven only through `R_index_pip_joint`, started from a clean reset with no initial contacts. Its operational OPEN driver target was `0.0015 rad`; the derived follower target was `0.0016455 rad`. It failed at step 3 (`t = 0.006 s`) during `open_hold_2s`, before the planned `+0.005 rad` step. The stop-on-first-failure rule therefore prevented the other representative cases from running.

At the failing sample, the tested follower was at `0.00328454 rad`, velocity `0.343648 rad/s`, acceleration `34.4457 rad/s^2`, and source relation residual `0.00351779 rad`. Its actual generalized actuator torque was `-0.000100667 N m`: PD `-0.00000654612 N m` plus bias feedforward `-0.0000941207 N m`; `qfrc_bias` was `-0.0000941207 N m`, with zero constraint force. The actuator used gear 1 and did not clip. Across the sample, maximum hand speed was `0.611353 rad/s`, maximum model acceleration was `157.00198 rad/s^2`, maximum target tracking error was `0.00279514 rad`, and maximum mimic residual was `0.00565659 rad`. The residual remained below the `0.010 rad` manipulation abort ceiling, but exceeding it was not the first failure.

Nine PIP joints crossed their source lower limit of zero: `L_index`, `L_middle`, `L_ring`, `L_pinky`, `R_thumb`, `R_index`, `R_middle`, `R_ring`, and `R_pinky`. The worst was `R_thumb_pip_joint` at `-0.000800143 rad`. There were zero velocity-limit violations, NaNs, contacts, self-contacts, effort-bound violations, or active/follower rollout qpos writes.

The harness records 54 pose-initialization qpos assignments before `t=0`; it records zero active or follower qpos writes after rollout start.

## One Bounded Diagnosis

The three-step startup diagnostic found all hand targets equal to their initial positions at `t=0`; PD error and PD torque were zero. Each hand actuator's bias feedforward equaled that joint's `qfrc_bias`, and the reconstructed MuJoCo dynamics equation residual was `8.88e-16 N m`. Despite that, peak initial hand acceleration was `157.00198 rad/s^2` at `R_index_abad_joint`.

Non-hand coordinates were not gravity-balanced or held at reset. Examples at `t=0` include `left_wrist_pitch_joint` (`qfrc_bias=-0.161836 N m`, actuator force `0`, acceleration `39.2653 rad/s^2`), `left_ankle_pitch_joint` (`-0.393131 N m`, `0`, `33.4620 rad/s^2`), `right_ankle_pitch_joint` (`-0.392974 N m`, `0`, `33.4476 rad/s^2`), and `right_elbow_joint` (`-1.09730 N m`, `0`, `28.6220 rad/s^2`). There were no constraint forces or contacts. This supports the diagnosis that free, unbalanced non-hand motion is coupling acceleration into the hand and carrying near-stop hand coordinates outside source ranges; it does not support attributing this first failure to the requested finger step or bottle/contact loads.

The smallest next correction is to establish a documented, dynamically supported no-contact whole-body initialization before revalidating this same first Stage 2 case. Keep source limits, the `0.0015 rad` target margin, the `0.05 N m` simulation bound, and the selected hand gains unchanged during that correction. No correction was applied in this task because the required one-diagnosis stop was reached.

## Stages Not Run

Stages 3–7 were not run. There was no 2-second full-hand OPEN pass, no full prevalidation, no fixed-object fingertip-contact test, and no bottle contact/hold/lift or repeatability test. No conclusion about manipulation readiness is supported.

## Evidence And Reproduction

The complete evidence packet is stored in `issue-46-x2-single-hand/virtual-transmission-20261007/timestep-aware-inertia-scaled-v1/` on the Windows-side review folder. It includes runtime identity, per-DOF controller design JSON, Stage 2 result and trace, the bounded startup diagnostic result and trace, and copied source harnesses with checksums.

Stage 2 command (from the RobotSim task checkout):

```bash
env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 \
  AGIBOT_X2_VENDOR_SHA=575cc6b988f976c23550e0db85aa1e5475d3652d \
  ROBOSIM_SHA=8ba8d241cdc0eff7c3ab59ba063aa22323b119c3 \
  ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/timestep-aware-inertia-scaled-v1 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_virtual_transmission_timestep_validation.py
```

The single bounded diagnosis used the same environment and interpreter, replacing the final script with `scripts/research/issue46_virtual_transmission_open_startup_diagnostic.py`. These are research-only harnesses. PR #53 remains Draft; no merge was performed.
