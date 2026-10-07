# Issue #46: X2 OmniPicker 1-DOF SIMULATION_ONLY_M0

Status: **FAIL - fixed-object CLOSE exceeded the paired-jaw tracking bound before bilateral contact**

## Outcome

The accepted Stage 3 OPEN hold and Stage 4 open-close-hold-reopen motion passed without contact. The fixture-only recovery replaced body-origin placement with an analytically derived pose from the compiled OPEN collision meshes. Its zero-step preflight passed with no initial robot-fixture contact or penetration. During the fixed-object CLOSE, the narrow jaw contacted the cylinder first, but the paired-aperture relation error exceeded its existing 0.020 rad diagnostic bound at step 850 / 1.700 s, before the wide jaw contacted it. The run stopped before bilateral contact, hold, or reopen. Bottle hold and lift were not run.

## Source and runtime identity

- Upstream: https://github.com/AgibotTech/agibot_x2_urdf
- Commit: 575cc6b988f976c23550e0db85aa1e5475d3652d
- URDF: X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf
- License: Mulan PSL v2
- URDF SHA256: 35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d
- Right driver: right_claw_joint, source range [-1, 0] rad
- Right follower: R_hand_wide1_joint, source range [0, 1] rad
- Both source axes: [0, 0, -1]
- Source relation: q_follower = -q_driver; multiplier -1, offset 0
- URDF transmission elements: 0
- Python: 3.10.12
- MuJoCo Python/native: 3.3.6
- Native library SHA256: b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495
- MUJOCO_GL: egl; NumPy 1.26.4
- Timestep: 0.002 s; integrator: Euler; solver: Newton; iterations: 100
- Accepted static scene helper SHA256: 41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae

## M0 command and controller

The only public jaw command is aperture_ratio in [0, 1], where 0 is CLOSED and 1 is OPEN. With endpoint margin m = 0.001 rad:

- q_driver(a) = -(m + (1 - 2m) * a)
- q_follower(a) = -q_driver(a)
- target velocities are the same signed mapping applied to aperture velocity
- OPEN at a=1: driver -0.999 rad, follower +0.999 rad
- CLOSED at a=0: driver -0.001 rad, follower +0.001 rad

The source hard ranges are unchanged. One shared minimum-jerk aperture reference is used for both targets, with 2.5 s movement duration. The controller uses joint-specific inertia-scaled critically damped gains:

| Jaw joint | kp (Nm/rad) | kd (Nm*s/rad) | effort cap (Nm) |
| --- | ---: | ---: | ---: |
| right_claw_joint | 0.005311594 | 0.001327899 | 0.238070571 |
| R_hand_wide1_joint | 0.010911462 | 0.002727866 | 0.253893293 |

Predicted peak joint speed is 0.7485 rad/s and predicted acceleration is 0.921913 rad/s^2. These are SIMULATION_ONLY_M0 engineering parameters, not hardware limits. The 0.02 rad paired-target tracking diagnostic is likewise an M0 engineering bound, not a source requirement.

The two jaw servos are internal implementation details driven from the shared aperture command. The right-jaw equality is absent. The left picker remains in a deterministic supported source-valid state and receives no contact or assistance. There were zero active-rollout qpos writes, zero follower qpos writes, and no object attachment.

## Stage results

| Stage | Result | Evidence |
| --- | --- | --- |
| 3 - OPEN hold, 2 s | PASS: 1000 steps; no source limit violation, contact, NaN, actuator saturation, or rollout qpos write; qvel settled; max relation error 0.000372644 rad | run2/open_hold_result.json and open_hold_trace.jsonl |
| 4 - open, close, hold, reopen, settle | PASS: 3125 steps / 6.25 s; no source limit violation, contact, NaN, actuator saturation, or rollout qpos write; max speed 0.748107 rad/s; max relation error 0.000372644 rad | run2/no_contact_motion_result.json and no_contact_motion_trace.jsonl |
| Fixture recovery - zero-step preflight | PASS: zero fixture contacts; no penetrating/touching collidable geoms; minimum collision-enabled clearance 1.004 mm at right_wrist_roll_link; intended narrow/wide jaw clearances 40.622 mm / 59.058 mm | 20261007-fixture-recovery-run1/fixed_object_result.json and fixed_object_open_fixture.png |
| Fixed-object CLOSE | FAIL: first contact at step 834 / 1.668 s on `R_hand_narrow3_Link` only; relation error reached 0.021374 rad at step 850 / 1.700 s against the unchanged 0.020 rad M0 bound; no bilateral frames; stopped before hold/reopen | 20261007-fixture-recovery-run1/fixed_object_result.json, fixed_object_trace.jsonl, and fixed_object_one_step_diagnosis.json |
| Bottle hold | NOT RUN | Fixed-object CLOSE failed before bilateral contact |
| 1 / 5 / 30 mm lift | NOT RUN | Bottle hold was not attempted |

The recovered fixed cylinder retained its original 20 mm radius and 25 mm half-height. Its X/Y center is the midpoint between the nearest compiled collision-surface witnesses on `R_hand_narrow3_Link` and `R_hand_wide3_Link`. Its vertical placement keeps the cylinder top 1 mm below the lowest vertex of the compiled right-wrist collision mesh. At the accepted OPEN pose, the actual target surface witnesses are 103.091 mm apart; the narrow and wide target geoms have 40.622 mm and 59.058 mm signed clearances to the fixture. The minimum collision-enabled clearance over the right hand/arm is 1.004 mm. No body origins or arbitrary XYZ offsets were used.

The first cylinder contact was on the narrow side only. Across the stopped run, maximum penetration was 0.416 mm, maximum measured normal force was 0.0775 N, maximum jaw speed was 0.7481 rad/s, maximum jaw acceleration was 75.6624 rad/s^2, and peak applied jaw efforts were 0.01141 Nm / 0.02506 Nm against caps of 0.23807 Nm / 0.25389 Nm. There were no source position violations, non-finite state, actuator saturation, or active-rollout qpos writes. These measurements do not establish stable bilateral contact. One bounded termination-state diagnosis is preserved; no further diagnosis or simulation was run.

The first open-hold attempt in run1 tripped a harness-only acceleration bound of 2.34383 rad/s^2 at step 1, observing 11.8322 rad/s^2. The bounded one-step diagnosis showed no contacts, finite state, no source-limit violation, and no actuator saturation; the later open-hold trace settled. That arbitrary bound was not a task requirement and was removed from the acceptance gate without changing targets or controller gains. The full 2 s open hold and full no-contact motion then passed in run2. Both raw attempts are retained.

## Evidence

External evidence root:

    /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-1dof-m0/

The run folders retain source audit, runtime identity, controller derivation, stage JSON, JSONL traces, the single termination-state diagnosis, and PNGs. The fixture OPEN close-up is `20261007-fixture-recovery-run1/fixed_object_open_fixture.png`. No MP4 was produced because the fixed-object CLOSE stopped before bilateral contact, hold, or reopen. Raw evidence was left unchanged.

The exact fixture-recovery command is preserved in `20261007-fixture-recovery-run1/experiment_commands.txt` and was:

    wsl.exe -d Ubuntu-22.04 -- bash -lc 'source /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/activate && python3 /tmp/robotsim-issue46-omnipicker-1dof-m0-20261007/scripts/research/issue46_omnipicker_1dof_m0.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-1dof-m0/20261007-fixture-recovery-run1 --stage-only fixed_object'

Use a fresh output directory when reproducing; the retained raw run directories should not be reused.

## Claim boundary

The X2 OmniPicker source geometry, joint ranges, and mimic mapping are source-derived. The paired-jaw actuator model is a SIMULATION_ONLY_M0 engineering abstraction and is not a validated model of the real X2 OmniPicker transmission or hardware dynamics.

No bottle-grasp or lift capability is established by these results.
