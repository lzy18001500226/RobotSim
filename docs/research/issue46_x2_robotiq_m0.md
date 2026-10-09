# Issue #46: X2 + Robotiq 2F-85 M0

Experiment outcome: FAIL at static preflight; physical manipulation was BLOCKED before physics. This is an isolated `SIMULATION_ONLY` experiment. It is not an original OmniPicker result or a hardware-validated X2 tool configuration.

## Outcome

Pinned sources compiled into a fixed-base X2 plus Menagerie Robotiq model. The gripper subtree mass matched the pinned Menagerie model (1.0526083388 kg), and the derived wrist-local mount reproduced the intended world jaw basis with 0 rad orientation error. No source or production asset was changed.

The initial collision-free gate failed in both tested station configurations. A 101 x 101 scan over the source head-yaw range [-0.488692, 0.488692] rad and head-pitch range [-0.401425, 0.488692] rad found zero collision-free samples for the source `head_pitch_link` cylinder (`size=0.08 0.04`, `pos=0 0 0.01`) against the `torso_link` collision mesh. The best sampled signed distance was -0.0061677 m at yaw 0 and pitch 0.435285 rad. The source MJCF has no explicit exclusion for this pair. Candidate 1 used base position [-0.08, 0, 0.68] m and had no gripper/table-leg overlap. Candidate 2 used [0, 0.08, 0.68] m and intersected the rear-left table leg with the Robotiq coupler (-0.012948 m), spring link (-0.014182 m), follower (-0.032070 m), and pads (-0.004287 m).

Source-limited pregrasp IK also failed in both candidates. The target was [0.18, 0, 0.8775] m. Candidate 1 residuals were 0.023042 m and 0.735764 rad; candidate 2 residuals were 0.019563 m and 0.533801 rad. The mount-frame consistency check passed, so these residuals remain an arm reach/orientation blocker rather than a wrist-local/world-frame mix-up.

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Pinned source identity and model compile | PASS | `result.json` identity and compiled model fields |
| Robotiq mounted on X2 with clear initial state | FAIL | `mounted_static` and `head_torso_clearance_audit` in `result.json` |
| Source-limited pregrasp IK | FAIL | `raw/ik_solver_trace.json` for both candidates |
| X2-mounted OPEN/CLOSE regression | NOT RUN | Initial collision-clearance gate failed |
| Physical bilateral bottle grasp | NOT RUN | No physics rollout started |
| Bottle load transfer | NOT RUN | No physics rollout started |
| Physical 30 mm lift | NOT RUN | No physics rollout started |
| Transfer, release, and placement | NOT RUN | No physics rollout started |
| Deterministic preflight reproduction | PASS | Same candidate residuals and clearance findings reproduced |

The first hard blocker is the source-model head/torso penetration. Candidate 1 avoids the table-leg collision but still fails that gate and IK. Candidate 2 adds table-leg penetration and still fails IK. No contact dynamics were run from these states.

## Provenance and Mount

- X2: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, Mulan PSL v2.
- X2 MJCF: `X2_URDF-v1.4.0/X2-Ultra.xml`, SHA-256 `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- Robotiq: `google-deepmind/mujoco_menagerie`, commit `0059d4335f8156206f63a35662313385f7ad6d74`, BSD-2-Clause.
- Robotiq MJCF: `robotiq_2f85/2f85.xml`, SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.
- Runtime: Python 3.10.12; MuJoCo Python/native 3.3.6; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Runner SHA-256: `dbca853b05e6c5b075c6f77cdfd2a7ab671b7e3dd4d0d103efaa12aa50d9c2b6`.
- Canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Translation uses the pinned X2 right tool-mount origin: `[0.00196, 0.00035, -0.062]` m.
- The adapter quaternion in the wrist frame is recorded in `result.json`. Its rotation was derived from the Robotiq insertion/pad axes and the source-neutral X2 wrist transform; it is an experimental adapter, not a vendor-validated flange interface.

The canonical 70 mm, 244.5 mm, 0.570 kg multipart bottle and table helper were kept unchanged, including bottle friction and gravity. The bottle remained a free-jointed body in the compiled model; no bottle attachment or active-rollout object-state write was used. Full X2 and Robotiq per-mesh hashes are in the `identity` section of `result.json`. Since no physics rollout started, this run has no physical contact trace or MP4.

## Evidence

External packet: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robotiq-x2-simulation-only-m0-20261009\`.

It contains the final `result.json`, compiled MJCF, static overview/front/side/mount screenshots, raw IK traces, and both bounded station candidates. Candidate 1 and candidate 2 have separate result and trace directories. Static images show the compiled assembly only; they are not manipulation evidence.

Reproduction commands and runtime prerequisites are in `REPRODUCE.md` in the same evidence packet.

## Next Decision

Do not start bottle contact or bypass the source collision. A maintainer decision is needed on how to handle the source head/torso collision pair and whether a different source-valid station/arm target is authorized. Any collision filtering or source model change requires explicit review before this experiment can proceed to dynamics.
