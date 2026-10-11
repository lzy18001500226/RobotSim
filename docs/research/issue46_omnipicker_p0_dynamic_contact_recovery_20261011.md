# Issue #46: OmniPicker Dynamic Contact Recovery

## Outcome

**FAIL — `LOAD_TRANSFER`: the fixed-wrist gripper formed sustained bilateral pad contact, but could not unload the table or carry the bottle during the commanded lift.** The furthest successful physical stage is the 1-second bilateral `HOLD`. The bottle did not lift; full-X2 integration was not run.

The tested branch was `research/issue46-omnipicker-p0-automation-20261011`, starting at `2960df261a1685e7dd57b1837171c540424334bd`. All experimental controls remain SIMULATION_ONLY paired-jaw actuator emulation. The accepted 11 historical simulation-only exclusions remain unchanged; no new exclusions were added.

## Diagnosis and correction

The previous run02 failure was weak closed-loop authority, not an effort-cap or bias-feedforward problem. At the CLOSE endpoint, driver/follower errors were `+19.182/-7.379 mrad`; peak requested/applied torques were only `0.000263/0.000446 N m` in CLOSE and `0.000202/0.000226 N m` in HOLD. Neither actuator saturated. Bias compensation stayed below `8.1e-8 N m`.

One controller correction was applied: simulation-derived stiffness now targets the recorded `3.994 N` design normal force per jaw at a `20 mrad` tracking deflection, with critical damping calculated from compiled joint inertia. Gains are `9.7783 N m/rad` for both joints; damping is `0.05697 N m s/rad` for `right_claw_joint` and `0.08166 N m s/rad` for `R_hand_wide1_joint`. Effort limits remain `0.224987/0.225078 N m`; actuator gear remains 1.0; bias sign and source follower mapping remain unchanged. This correction reduced CLOSE endpoint error to `+5.998/-3.699 mrad` and produced 100% bilateral contact throughout HOLD.

The remaining failure is inadequate load-bearing normal preload at the accepted contact endpoint. During HOLD, mean normal force was only `0.4335 N` on `R_hand_narrow3_Link` and `0.3912 N` on `R_hand_wide3_Link`; the table carried an average `5.592 N` while mean hand vertical support was effectively zero. During the 50 mm carriage command, bilateral contact persisted for 90.2% of samples, but mean hand vertical support was `0.693 N` (peak `1.412 N`) and mean table support remained `4.898 N`. The bottle stayed on the table and slipped approximately `50.0 mm` relative to the rising gripper.

This was not caused by actuator saturation or unstable tracking: during HOLD the peak torques were `0.061642/0.045477 N m`, below their unchanged caps, with zero saturated samples; maximum post-step tracking errors were `6.103/4.217 mrad`. Static geometry at the selected bilateral target reports `1.728 mm` narrow-pad and `0.508 mm` wide-pad overlap; dynamic contact penetration remained at or below `0.696 mm`. The static and dynamic penetration measures are distinct. No second correction was applied: the current data does not identify a deeper target that can add the missing preload while respecting the existing dynamic penetration cap.

## Run comparison

| Measure | Previous run02 | Corrected run03 contact-only | Corrected run04 lift attempt |
|---|---:|---:|---:|
| CLOSE endpoint error, driver / follower | `+19.182 / -7.379 mrad` | `+5.998 / -3.699 mrad` | `+5.998 / -3.699 mrad` |
| CLOSE peak requested/applied torque, driver / follower | `0.000263 / 0.000446 N m` | `0.070728 / 0.040242 N m` | `0.070728 / 0.040242 N m` |
| HOLD end error, driver / follower | `+18.856 / -6.183 mrad` (max abs `19.476 / 7.379`) | `+3.547 / -3.592 mrad` (max abs `6.103 / 4.217`) | `+3.547 / -3.592 mrad` (max abs `6.103 / 4.217`) |
| HOLD peak torque request/applied, driver / follower | `0.000202 / 0.000226 N m` | `0.061642 / 0.045477 N m` | `0.061642 / 0.045477 N m` |
| Effort caps, driver / follower | `0.224987 / 0.225078 N m` | unchanged; zero saturation | unchanged; zero saturation |
| HOLD bilateral contact fraction | `0.2%` | `100%` | `100%` |
| HOLD pad normal force, narrow / wide | intermittent, peak `0.0365 / 0.0706 N` | mean `0.4335 / 0.3912 N`, peaks `0.6251 / 0.4736 N` | mean `0.4335 / 0.3912 N`, peaks `0.6251 / 0.4736 N` |
| Maximum bottle lift | `0 mm` | no lift commanded | `0 mm` |

Run04 lift stages confirmed failure to transfer support. At `LIFT_1MM`, table mean support was `5.511 N`; at `LIFT_5MM`, `5.372 N`; at `LIFT_50MM`, `4.898 N`, versus bottle weight about `5.592 N`. At `HOLD_50MM`, table support returned to `5.577 N`. Maximum bottle bottom height during the 50 mm stage remained between `0.799353` and `0.799442 m`; the table top is `0.800 m`. Maximum hand support across that stage was `1.412 N`. Maximum relative slip was `0.050023 m`.

For both corrected runs, `actuator_force` and generalized actuator torque equal requested/applied torque because the motor gear is 1.0; no clamping occurred. Peak `qfrc_bias` was negligible (below `8.1e-8 N m` in CLOSE/HOLD). Thus feedforward sign or a gear double-application does not explain the load-transfer failure.

## Integrity and validation

- Static source-collision corridor: PASS; minimum non-pad clearance `4.498 mm`, table clearance lower bound `75.489 mm`.
- Moving-fixture no-contact regression: PASS, 2,250/2,250 steps; zero source-limit violations, no nonfinite state, no unexpected robot contacts.
- Contact-only run03: PASS for one-second bilateral HOLD; no lift commanded.
- Lift run04: FAIL at load transfer; zero bottle rise. Full X2 integration: NOT RUN.
- Source joint ranges stayed `right_claw_joint [-1,0] rad` and follower `[0,1] rad`; no limit violations.
- No bottle qpos writes, weld/equality attachment, mocap carry, artificial force, hidden support, source geometry change, or new collision exclusion.
- Maximum recorded contact penetration was `0.696 mm`; no NaNs.
- Maximum mimic residual during the corrected contact/lift run was `0.005759 rad`, below the existing `0.020 rad` simulation diagnostic bound.
- Both videos were decoded successfully with the pinned environment FFmpeg binary.

## Reproduction and evidence

Source: AgiBot X2 URDF commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`. MuJoCo Python/native `3.3.6`, Python `3.10.12`, timestep `0.002 s`; native library SHA256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`. Generated right-wrist subtree SHA256 `20d5221580c003e8d05e3f35985a75a61c25f8f81483b82f7b41a878c9c82786`.

Tested runner SHA256: `cbfd9225c472fab7e5b87072c38abf85175c9b817040bb7ff45df7dcdaecdd01`.

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011
PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 MUJOCO_GL=egl python3 scripts/research/issue46_omnipicker_p0_isolated.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-p0-open-endpoint-20261011-run03 --contact-only
```

The lift attempt used the same command with the `--contact-only` flag removed and the run04 output path:

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011
PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 MUJOCO_GL=egl python3 scripts/research/issue46_omnipicker_p0_isolated.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-p0-open-endpoint-20261011-run04
```

Existing raw evidence was preserved:

- Contact-only result/video/trace: `C_RESULT.json`, `C_bilateral_contact_hold.mp4`, `C_CONTACT_HOLD_TRACE.jsonl` in `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-p0-open-endpoint-20261011-run03\`.
- Lift attempt result/video/trace: `C_RESULT.json`, `C_isolated_contact_lift.mp4`, `C_CONTACT_LIFT_TRACE.jsonl` in `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-p0-open-endpoint-20261011-run04\`.

The full-X2, release, and placement stages remain unverified. No changes were made to PR #58.
