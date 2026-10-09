# Issue #46: X2 + Robotiq M0 Station Recovery

**Status: BLOCKED BEFORE BOTTLE DYNAMICS.** This is a `SIMULATION_ONLY` experiment. The head/torso exclusion is an explicit modeling exception, and no result here establishes source-faithful hardware behavior.

This follow-up preserves the earlier static-preflight report and its evidence. It compares the pinned source and mounted model, validates the authorized exact-pair exclusion, runs the mounted OPEN/CLOSE cycle, then stops after two failed source-limited static approach configurations. Bottle contact, lift, transfer, and release were not run.

## Gate Results

| Gate | Result | Finding |
|---|---|---|
| Source versus mounted head/torso | PASS | Both models have the same sole initial self-contact, `head_pitch_link` ↔ `torso_link`, at `-0.007112461424559683 m`. Geometry world transforms and contact position match exactly. The pinned source has no explicit exclusion. |
| Authorized exact-pair exclusion | PASS, diagnostic only | The separate experimental XML adds only `head_pitch_link` ↔ `torso_link`; no source geometry, masks, limits, or other pairs changed. The excluded model has zero initial active contacts. |
| Mounted dynamic OPEN/CLOSE | PARTIAL; hard-limit audit FAIL | The gripper opens, closes, and reopens with no robot self-contact, robot/bottle contact, velocity-limit violation, or bottle qpos write. Both Robotiq coupler joints exceed their unchanged upper limit by up to `0.000150054301 rad`, beginning at step 1. |
| Revised source-limited PREGRASP/APPROACH | FAIL | Neither of the two authorized configurations reached a collision-free full-pose pregrasp within the 3 mm / 0.02 rad pose tolerances. Position-only solutions intersect the table substantially. |
| Bilateral physical bottle contact | NOT RUN | Static approach gate failed. |
| Load transfer and 30 mm lift | NOT RUN | Static approach gate failed. |
| Transfer and physical release | NOT RUN | Static approach gate failed. |

No bottle manipulation was attempted. The overall result is **BLOCKED**, not a grasp failure and not evidence of global X2/Robotiq infeasibility.

## Source and Contact Exception

At the same source-valid neutral head pose (`head_yaw=0`, `head_pitch=0`; elbows `-0.01 rad`) and station base `[-0.08, 0, 0.68] m`:

- Pinned source `X2-Ultra.xml`: one active self-contact, source geom IDs 94/61, distance `-7.112461424559683 mm`.
- Mounted X2 + Robotiq: the same one active self-contact, mounted geom IDs 123/62, the same distance and contact point.
- Head and torso geometry world-position and rotation deltas are zero; contact-position max absolute delta is zero.
- Source head limits are unchanged and satisfied. No second major initial collision was observed.
- The diagnostic variant adds exactly `<exclude name="issue46_simonly_head_pitch_torso" body1="head_pitch_link" body2="torso_link"/>`. It retains the original geoms and all other collision pairs; active initial contact count becomes zero.

The recorded pre-step contact-force estimate is from `mj_forward` on the penetrated pose, not a measured rollout force. The exclusion remains a known `SIMULATION_ONLY` exception and is not source-faithful validation.

## Mounted Gripper Cycle

The 1,900-step, 1.9-second MuJoCo run completed `INITIAL HOLD → OPEN → CLOSE → REOPEN → HOLD`:

- Pad separation: `93.162682 mm` open, `14.690997 mm` closed, `86.245835 mm` after reopen.
- Driver `rq_right_driver_joint` reached `0.727793652 rad`; paired follower ended at `0.008818402 rad`, matching the driver at final hold within the measured trace precision.
- Peak absolute gripper actuator force: `0.676788183`, below the configured `5` force limit.
- X2 joints had no source position or velocity violations; maximum non-gripper deviation was `0.0000216374 rad`.
- No robot self-contact samples, no robot/bottle contact samples, and zero post-start bottle qpos writes. The bottle remained on its table support.
- `rq_right_coupler_joint` peaked at `+0.000150054301 rad` and `rq_left_coupler_joint` at `+0.000149957023 rad`, both against source/compiled range `[-1.57, 0] rad`; both first crossed the upper limit at step 1. The limit was not changed.

Thus the jaw motion itself is demonstrated, but the full dynamic gate is not a clean PASS because the coupled-jaw hard-limit audit fails. No controller correction was attempted.

## Bounded Approach Configurations

Both configurations used the compiled bottle-body geom center `[0.3, 0, 0.8775] m`, a `0.12 m` pregrasp offset, and the same right-side insertion direction `+Y` (tool yaw `90 deg`). The target was compared against compiled `rq_m0_tcp` FK; source joint limits were enforced in every solve.

| Candidate | Fixed base (m) | Collision-aware full-pose result | Position-only FK diagnostic |
|---|---|---|---|
| 1, right-side approach | `[-0.08, 0, 0.68]` | Best recorded full-pose result: `19.0364 mm` position error and `1.02293 rad` orientation error, with no active contact. Fails both pose tolerances. | Pad midpoint reaches target within `4.13e-14 m`, but table contacts include wrist `-65.557 mm`, base mount `-21.625 mm`, and base `-15.337 mm`. |
| 2, lateral station shift | `[-0.08, 0.08, 0.68]` | Best recorded full-pose result: `234.632 mm` position error and `0.441771 rad` orientation error, with no active contact. Fails pose tolerances. The alternate position-seeded full solve has bottle/table contacts and is also invalid. | Pad midpoint reaches target within `2.03e-16 m`, but table contacts include wrist `-76.393 mm`, base mount `-34.530 mm`, and base `-53.596 mm`. |

The solver audit identified why the first pass was misleading: its wrist-only orientation seed rotated the mounted gripper into the hip before the full solve. The corrected runner preserves that seed as diagnostic evidence and starts full-pose IK from the verified clear neutral pose; it also tests a bounded position-only FK seed before a collision-aware full-pose solve. The corrected seeds did not produce a valid target pose. These results indicate the tested centered bottle-height/90-degree approach is blocked by pose and table-clearance constraints; they do not prove other source-valid grasp heights or orientations impossible.

The current task's two-config budget is exhausted. No additional pose sweep, station search, or contact tuning was performed.

## Provenance

- RobotSim runtime HEAD: `3ed4c3aeace7907d4c7a3d2c938f3727dbe3b1c2`; research runners were untracked at execution and are identified by content hashes below.
- X2: `AgibotTech/agibot_x2_urdf`, pinned `575cc6b988f976c23550e0db85aa1e5475d3652d`, clean checkout, Mulan PSL v2.
- X2 MJCF: `X2_URDF-v1.4.0/X2-Ultra.xml`, SHA-256 `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- Menagerie: `google-deepmind/mujoco_menagerie`, pinned `0059d4335f8156206f63a35662313385f7ad6d74`, clean checkout; `robotiq_2f85/2f85.xml` SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`, BSD-2-Clause.
- Runtime: Ubuntu 22.04 / WSL2, Python `3.10.12`, MuJoCo Python/native `3.3.6`, `MUJOCO_GL=egl`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Mounted smoke runner SHA-256: `92d6b5d4f610fcb8574f0527309d8326df6bc1c4b88de45e9fc2b3b1c2b8e4c1`.
- Reachability runner revisions: initial diagnostic seed `0602504411b3d4c8cf6cf963d679ba9e3565a33d19f5bb390b26391d20076fdf`; neutral-pose seed correction `fd5cc76868a4d1dc989c05493bb8c301c82ee31f4a2b738a4f7fc03334b36e94`; position-only FK seed plus collision-aware full-pose solve `f2f5ec36f6efbde998f076407e8d2174723f39f24c5b789d7e0aa765307cb933`.
- Base M0 builder SHA-256: `dbca853b05e6c5b075c6f77cdfd2a7ab671b7e3dd4d0d103efaa12aa50d9c2b6`.
- Candidate 1 diagnostic model SHA-256: `815a89f421871f396760391d11a50d165b189965e8204606dbb76b0eb66bc1b3`.
- Candidate 2 diagnostic model SHA-256: `878dd844a27359335d0360de902a014ff763a2ae048dd72464fbb73abd1def72`.

## Evidence and Reproduction

The self-contained review packet is at `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robotiq-x2-m0-20261009-final\`. It contains copies of the report, result JSON, reproduction commands, runner scripts, and complete attempt/run folders under `evidence/`. The original raw capture folders remain intact and were not rewritten:

- Mounted contact comparison, exact exclusion XML, 1,900-step trace, contact trace, result JSON, and MP4: `...\robotiq-x2-head-contact-smoke-20261009\attempt_04\`.
- Candidate runs 1–3, exact joint poses, optimizer traces, compiled model XMLs, logs, and PNGs: `...\robotiq-x2-reachability-20261009\run_01\`, `run_02\`, and `run_03\`.

See [`issue46_x2_robotiq_m0_station_recovery_20261009_reproduce.md`](issue46_x2_robotiq_m0_station_recovery_20261009_reproduce.md) for exact commands. `SHA256SUMS.txt` in the external packet covers the packet contents, including the copied raw evidence. The head/torso exclusion affects only that named internal pair; the mounted gripper cycle ran with the head stationary, and no other self-collision pair was excluded.

## Stop Point

Bottle contact and all later physical manipulation gates remain unrun. Further work needs a new bounded maintainer decision on a materially different, geometry-derived target (for example grasp height or approach direction) and disposition of the Robotiq coupler endpoint transient. Issue #46 remains open; PR #58 remains Draft.
