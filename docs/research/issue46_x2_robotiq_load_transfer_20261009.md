# Issue #46: X2 + Robotiq Isolated Load Transfer

## Result

**FAIL — `BOTTLE_NOT_AIRBORNE`.** The final bounded attempt reached a transient 1.168 mm bottle COM rise and ended the hold at 1.158 mm, but the bottle recontacted the table during 28.5% of the 0.2 s hold. The source-faithful Robotiq model also violates both coupler upper limits on the first physics step. No accepted lift milestone was passed, release was not attempted, and full X2 arm integration was not run.

This is an isolated MuJoCo contact experiment, not hardware validation. The later grasp-height and carriage-controller candidates are explicitly SIMULATION_ONLY. A transient bottle rise during a move is not counted as an airborne held lift.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| Static fixture preflight, COM-aligned pad midpoint | PASS | 7.7 mm minimum pad clearance; 13.629 mm minimum non-gripping robot-to-bottle clearance; 55.000 mm minimum robot-to-table clearance; no initial unintended contacts or source joint-limit violations. |
| Bilateral physical contact and 1 s hold | Contact PASS; source-limit gate FAIL | Both pad families remained in contact for 100% of 1,000 hold samples. Bottle XY COM slip was 3.83 µm. The table continued to carry most of the bottle load. |
| Full table load transfer | FAIL | At the final 1 mm hold, pads applied +5.131 N vertical force to the bottle and the table still applied +0.458 N. The table contacted the bottle during 57/200 hold samples. |
| 1 mm fully airborne hold | FAIL | Peak COM rise during motion was 1.168 mm; end-of-hold rise was 1.158 mm, but table contact persisted for 28.5% of the hold. The milestone requires no table contact for the whole hold. |
| 5 / 30 / 50 mm held lift | NOT RUN | The runner stopped at the first failed 1 mm airborne gate. |
| Physical release and free settle | NOT RUN | Release is gated on passing the 50 mm hold. |
| Full X2 arm integration | NOT RUN | The isolated load-transfer prerequisite did not pass. |

## Load-Transfer Diagnosis

The instrumented force balance explains the original failure. With the original 1,200 N/m carriage servo and no bias compensation, the carriage did not rise to the 1 mm target; its actuator force was about 1.53 N against about 10.33 N of gravity bias, and bottle rise was only 0.167 mm. During the original 1 s hold, pad vertical force on the bottle averaged -2.537 N while table support averaged +8.129 N. High opposing jaw-normal forces alone did not establish vertical load support.

A simulation-only carriage bias feedforward improved tracking. A 5x stiffness candidate improved load transfer but left table support. The 25x candidate was evaluated with the damping ratio preserved and the 25 N force cap unchanged. A separately validated +15 mm pad-midpoint height offset aligned the grasp plane with the bottle COM region without changing the bottle or gripper geometry. The last command used a 1.30 mm carriage target. At the endpoint, the carriage was at 1.128 mm, bottle COM rise was 1.158 mm, pads supported 5.131 N vertically, and the table still supported 0.458 N. The table gap closed to roughly 34 µm and recontact occurred during the hold. Bilateral pad contact persisted throughout; no non-pad robot geom contacted the bottle in this final candidate.

An earlier, less constrained 30 mm exploratory trajectory reached 24.4 mm transiently, then the tilted bottle struck the Robotiq base with about 28.7 N and lost height. It is not a held lift and is not an acceptance result. The hardened runner now stops at the first 1 mm airborne failure and does not continue to larger lift commands.

## Coupler Source-Limit Diagnosis

The pinned Robotiq model initializes both couplers at `qpos=0`, exactly at the source upper limit `[-1.57, 0] rad`. Each spring-link joint initializes at `qpos=0` with source stiffness `0.05 N·m/rad` and spring reference `2.62 rad`; this yields `+0.131 N·m` passive spring torque at reset. With no contacts and no actuator force, the active closed-loop equalities transmit a positive reaction to each coupler. At reset, each coupler has approximately `+0.012384 N·m` constraint force and `+12.090 rad/s^2` acceleration. After the first 1 ms step each is at `+1.209308e-5 rad`, outside the unchanged upper limit. The maximum observed excursion in the final run was `+0.000558411 rad` for the right coupler at step 787; both sides violated the limit.

One isolated in-memory causal diagnostic changed only the two spring stiffness values to zero. With no contacts, coupler acceleration fell from `+12.090 rad/s^2` to approximately `-2.19e-11 rad/s^2`, and the first-step coupler position remained at numerical zero. This establishes that the spring preload, transmitted through the closed-loop constraints while reset sits on the hard stop, causes the first-step violation. This altered model is **SIMULATION_ONLY DIAGNOSTIC ONLY**; it is not a source-faithful solution and was not used for a bottle rollout. Source limits and the pinned source file were not changed.

The remaining source-faithful blocker is therefore a reset/preload/limit equilibrium conflict in the pinned model. The run does not establish that a particular spring or constraint change is physically correct. Vendor evidence or maintainer direction is needed before changing that physical model.

## Integrity and Model Boundary

- Canonical bottle remains 70 mm diameter, 244.5 mm total height, 0.570 kg.
- Robotiq source mesh, joints, equality topology, joint ranges, actuator, pad friction, and required contacts were not changed.
- No bottle qpos writes, weld/equality bottle attachment, mocap, follow-hand logic, artificial bottle force, hidden support, or disabled contacts were used.
- The carriage is a simulation fixture. Bias compensation, its 25x stiffness profile, the 1.30 mm carriage target, and the +15 mm pad-midpoint scene offset are simulation-derived diagnostics, not source hardware parameters.
- No X2 arm, station, or IK search was run.

## Provenance

- Repository branch: `research/issue46-x2-robotiq-m0-20261009`.
- Reviewed repository HEAD before edits: `4e0c47042d287883293cd58f5d371252188fdc51`.
- Final verified runner SHA-256: `25b5eee232d5a74ef333e9d2fdeefee345b5b71fcffe367e0a439d33d52a887b`.
- MuJoCo Menagerie pin: `0059d4335f8156206f63a35662313385f7ad6d74`.
- Source model: `robotiq_2f85/2f85.xml`, SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.
- Canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Final compiled fixture SHA-256: `01a3f1c13b1d6de5d94dcf9e616ab91f3a4a348beeea5fab1e08e1cd77eb654b`.
- Runtime: Python 3.10.12, MuJoCo Python/native 3.3.6, `MUJOCO_GL=egl`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- External evidence root: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robotiq-load-transfer-20261009\`.

Raw per-run `result.json`, physics JSONL, contact JSONL, MP4, static images, and generated model XML are preserved under that evidence root. The corrected final candidate is `grasp_com_z15_clearance13_verified/`; its `run.log` and result were collected from the same runner SHA above. The initial baseline's shell redirection was attempted before its parent log directory existed; its result JSON records the exact argv and its raw traces and media are present, but that baseline has no `run.log`. No baseline log was reconstructed or fabricated.

## Reproduction

See [the reproduction instructions](issue46_x2_robotiq_load_transfer_20261009_reproduce.md). The corrected final candidate's result and raw traces are under `grasp_com_z15_clearance13_verified/`; the preceding run remains preserved under `grasp_com_z15_clearance13/`. Use a fresh output directory when reproducing so the preserved evidence is not overwritten.
