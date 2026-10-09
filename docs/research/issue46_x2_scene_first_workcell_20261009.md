# Issue #46: Scene-First X2 + Robotiq Workcell

## Outcome

The isolated SIMULATION_ONLY Robotiq test independently verifies a 50.189 mm bottle COM lift and a one-second bilateral, table-free hold. A mounted fixed-base X2 + Robotiq run then achieved a 56.295 mm peak bottle COM lift, held at least 54.224 mm for one second, returned to the starting position, released the bottle, and allowed it to settle under free physics. The mounted result is `PARTIAL_PASS`, not a complete pick-and-place: the source-limited transfer/place IK endpoint gates failed, so transfer to the target and placement there were not run.

The mounted run uses two explicitly SIMULATION_ONLY treatments: a 1 mm coupler-limit activation margin and one exact `head_pitch_link`/`torso_link` contact exclusion. Source joint limits, X2 and Robotiq meshes, bottle dimensions/mass/friction, gravity, and contact geometry were not changed. The source-faithful Robotiq coupler-limit path remains FAIL; this experiment is not source-faithful or hardware validation.

## Preserved Isolated Pickup

Independent verification is recorded in `isolated_lift_independent_verification.json`. It confirms:

- Peak bottle COM lift: `0.050188827 m`.
- Bilateral pad contact: all 1,000 hold samples over `0.999 s`.
- Table and non-gripping contacts during hold: zero.
- Post-start bottle qpos writes: zero; joint-limit violations: zero.
- Hold XY slip: `0.00002173 m`.
- Compiled bottle/table clearance: `0.0495166 m`.
- Classification: `ISOLATED_SIMULATION_ONLY; not source-faithful acceptance`.

## Mounted Workcell and Runtime

- Table center `(0.30, -0.10) m`; canonical table top at `z=0.80 m`.
- Bottle root `(0.30, 0.045, 0.9175) m`, 20 mm inside the table's north edge.
- Fixed X2 base `(0.30, 0.40, 0.68) m`, yaw `-90 deg`, facing the table.
- Target bottle root for the unexecuted transfer `(0.30, 0.020, 0.9175) m`.
- Python `3.10.12`; MuJoCo Python/native `3.3.6`; `MUJOCO_GL=egl`; timestep `0.001 s`.
- X2 source `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`; source MJCF SHA256 `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- Robotiq source `google-deepmind/mujoco_menagerie`, commit `0059d4335f8156206f63a35662313385f7ad6d74`, model `robotiq_2f85/2f85.xml`.
- Native MuJoCo library SHA256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.

The original X2 model has an internal neutral-pose head/torso penetration (about `7.1 mm`). The mounted diagnostic model preserves both bodies and collision geoms and excludes only that exact pair. No other robot collision pair is excluded. The diagnostic coupler margin is simulation-derived and does not alter source joint ranges. The original source-faithful coupler model still fails at reset due to its coupler spring torque exceeding the source stop.

## Decision Record

| Attempt | First blocking evidence | Formulation decision |
|---|---|---|
| Layout 1, oblique edge | Initial grasp IK residual `12.605 mm` and `0.5096 rad`; after a grasp-first formulation, the raised grasp still put `right_wrist_roll_link` `5.344 mm` into the table, and the lower target increased penetration to `15.127 mm`. | Moved the fixed base and bottle to the north table edge and faced the X2 toward the table. No claim of global infeasibility. |
| Layout 2, 30 mm grasp-height/retreat variants | The lower target collided between wrist and table by `4.949–5.344 mm`; an early 20 mm-retreat run also crossed the right shoulder-roll upper source limit by about `2.66e-6 rad`. | Used the compiled gripper/TCP frame, a 50 mm grasp-height offset, and a 20 mm pregrasp retreat; source limits remained unchanged. The final open approach and grasp pose were collision-free. |
| Layout 2, 55 mm mounted lift | At the bottle-height peak, the commanded 55 mm TCP path produced `42.024 mm` actual TCP travel and `31.168 mm` bottle COM travel from lift-path start. The bottle-to-TCP vertical gap increased `10.856 mm`; bilateral contacts persisted. The one-second hold minimum was `28.988 mm`. | A 50 mm bottle rise therefore required about 60.856 mm of TCP rise. The observed TCP tracking ratio `42.024/55 = 0.7641` implies a command near `79.65 mm`; the next run used 80 mm, with the same geometry, contact properties, and controller. |
| Layout 2, 80 mm lift | Source-limited IK and a 1,616-sample static loaded lift corridor passed. Bottle COM rose `56.295 mm`; one-second hold minimum was `54.224 mm`; the bottle was returned and released. | Pickup milestone passed in the mounted SIMULATION_ONLY model. Transfer remains blocked by endpoint IK, not by a tested collision corridor. |

## Mounted 80 mm Trial

- Mounted static initialization and self-contact gate: PASS; no initial contacts.
- Source-valid grasp and 20 mm pregrasp: PASS; open approach sampled at 101 collision-free points.
- Dry mounted gripper open/close/reopen: PASS; jaw gap `93.40 mm` open to `11.54 mm` closed; no bottle contact in the dry regression.
- Bilateral grasp/hold: PASS; bilateral pad contact in `100%` of hold samples for `1.0 s`; maximum bottle slip during the hold `0.236 mm`.
- Source-limited 80 mm Cartesian lift: all 16 IK waypoints passed at 5 mm spacing; loaded static lift corridor passed 1,616 samples.
- Physical lift: peak bottle COM rise `56.295 mm`; table-unsupported lift samples `675`; bilateral pad contact `100%` of the lift.
- One-second airborne hold: minimum COM rise `54.224 mm`; bilateral pad contact `100%`; table normal force `0 N`; maximum bottle slip `2.007 mm`.
- Contact families: bottle against `rq_right_pad1` and `rq_left_pad1`; occasional `rq_left_pad2` contact. No opposite hand contact or bottle attachment.
- Combined recorded pad normal force peaked at `133.06 N` during the hold. Maximum recorded bottle contact penetration magnitude was `1.219 mm`.
- Return/release: PASS at the starting location; pad contact was lost and the bottle settled freely for one second. Final linear speed `0.00219 m/s`. This is not placement at the transfer target.
- Bottle freejoint: true; no bottle weld/equality, mocap/follow, hidden support, opposite-hand assistance, or post-initialization bottle qpos writes.

## Remaining Blocker

The `25 mm` transfer target toward table center was not executed. Source-limited IK residuals were `3.362 mm / 0.05018 rad` for transfer, `6.128 mm / 0.09152 rad` for place, and `19.037 mm / 0.28530 rad` for retract. The loaded transport corridor was therefore not evaluated. The next change should alter the transfer/release geometry or station formulation, then check a source-valid endpoint and loaded path before another physical run. Do not claim a complete transfer, target placement, or full X2 pick-and-place.

## Evidence

Raw runs and screenshots are preserved externally at:

`C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\scene-first-workcell-20261009\`

The final mounted run is `layout_02_north_edge_80mm_evidence_derived_lift/`; the prior 55 mm run and the independently verified isolated lift are separate and unchanged. `REPORT.md`, `REPRODUCE.md`, and `SHA256SUMS.txt` in that evidence directory identify the exact command and raw artifacts. PR #58 was not modified.
