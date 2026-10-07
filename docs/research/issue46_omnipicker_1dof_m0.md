# Issue #46: X2 OmniPicker 1-DOF SIMULATION_ONLY_M0

Status: **FAIL - Stage 5 fixed-object initial-penetration preflight**

## Outcome

Stages 3 and 4 passed. Stage 5 stopped before its first physics step because the diagnostic cylinder intersects the wrist and multiple picker links at initialization. The final bounded fixture correction reduced the wrist overlap only from 34.261 mm to 33.858 mm. No fixed-object contact rollout, bottle hold, or lift was attempted.

The contacts reported at time zero are from an invalid overlapping initial geometry. Their solver force values are not evidence of stable grasp or contact behavior.

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
| 5 - fixed-object contact | FAIL before stepping: 0 steps; max initial penetration 0.0338583 m at right_wrist_roll_link; other overlaps at R_hand_narrow1_Link (0.00241190 m), R_hand_narrow_loop_Link (0.0185614 m), and R_hand_wide_loop_Link (0.00819942 m) | run3/fixed_object_result.json and fixed_object_trace.jsonl |
| 6 - bottle hold | NOT RUN | Stage 5 failed |
| 7 - 30 mm lift | NOT RUN | Stage 5 failed |

Stage 5 had one bounded geometry correction after the first preflight: the fixture cylinder half-height was reduced and its center lowered. The rerun still intersects the wrist and loop links, so the experiment stops here without further placement tuning.

The first open-hold attempt in run1 tripped a harness-only acceleration bound of 2.34383 rad/s^2 at step 1, observing 11.8322 rad/s^2. The bounded one-step diagnosis showed no contacts, finite state, no source-limit violation, and no actuator saturation; the later open-hold trace settled. That arbitrary bound was not a task requirement and was removed from the acceptance gate without changing targets or controller gains. The full 2 s open hold and full no-contact motion then passed in run2. Both raw attempts are retained.

## Evidence

External evidence root:

    /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-1dof-m0/

The run folders retain source audit, runtime identity, controller derivation, stage JSON, JSONL traces, one-step diagnosis, and no-contact PNGs. No MP4 was produced because the fixed-object stage failed before rollout. The no-contact PNGs were generated but were not visually reviewed in this closeout.

The exact commands used are in the external run folders' experiment_commands.txt. The full pipeline command in run2 reached the first failed stage. The final bounded Stage 5 rerun command in run3 was:

    wsl.exe -d Ubuntu-22.04 -- bash -lc 'source /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/activate && python3 /tmp/robotsim-issue46-omnipicker-1dof-m0-20261007/scripts/research/issue46_omnipicker_1dof_m0.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-1dof-m0/20261007-run3 --stage-only fixed_object'

Use a fresh output directory when reproducing; the retained raw run directories should not be reused.

## Claim boundary

The X2 OmniPicker source geometry, joint ranges, and mimic mapping are source-derived. The paired-jaw actuator model is a SIMULATION_ONLY_M0 engineering abstraction and is not a validated model of the real X2 OmniPicker transmission or hardware dynamics.

No bottle-grasp or lift capability is established by these results.
