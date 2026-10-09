# Issue #46: Isolated Robotiq Lift and X2 Integration Gate

## Outcome

**BLOCKED — integrated X2 source-limited PREGRASP was not reached.** The isolated Robotiq 2F-85 diagnostic physically lifted the canonical bottle by 50.189 mm and held it airborne for 1 s. Three bounded, collision-aware X2 integration candidates then failed the static pregrasp/alignment gate. No X2 approach, bottle contact, or integrated lift physics ran. This is not a complete X2 grasp and does not establish global X2 reachability failure.

The isolated result is explicitly **SIMULATION_ONLY**. Its carriage controller, 1 mm coupler-limit activation margin, target offsets, and carriage commands are simulation-derived. Source joint range values were not changed; the run recorded no state outside those ranges. Because constraint activation and fixture control differed from the pinned source-faithful model, this is not a source-faithful acceptance result.

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Isolated OPEN/CLOSE and bilateral contact | PASS | Both Robotiq pad families contacted the canonical 70 mm, 0.570 kg bottle. The initial HOLD persisted for 1 s. |
| Isolated load transfer and 1/5/30/50 mm lifts | SIMULATION_ONLY PASS | The final 50 mm move reached 50.189 mm maximum bottle COM rise. The 30 mm and 50 mm 1 s holds each had bilateral pad contact in 1000/1000 samples and table contact in 0/1000 samples. |
| Release and free physics | Release observed; stable rest not established | Opening ended with zero pad contacts. The bottle returned to table support during the 1 s free-physics segment, but ended with nonzero linear and angular velocity; do not call this a settled placement. |
| Integrated X2 static pregrasp | FAIL for all three bounded candidates | No candidate passed source-limited pregrasp, open-grasp alignment, 30 mm TCP lift, or the complete static corridor gates. |
| Integrated X2 physics | NOT RUN | Static geometry did not pass; no dynamic X2 approach or bottle contact was attempted. |

### Isolated Physical Record

- Bottle: canonical cylindrical body, 70 mm diameter, 0.570 kg; geometry and physical properties unchanged.
- Maximum measured COM lift: `0.05018882745320752 m`.
- At the 50 mm hold endpoint: bilateral pad contact `1000/1000`, table contact `0/1000`, non-gripping bottle contact `0/1000`.
- At the 30 mm hold endpoint: bilateral pad contact `200/200`, table contact `0/200`.
- Bottle qpos writes after rollout start: `0`; bottle attachment or mocap: `false`.
- Both Robotiq pads contacted the bottle; no other robot geom contacted it during the lift holds.
- The coupler `jnt_range` values remained `[-1.57, 0] rad`; the run reported source-range checks PASS. A simulation-only `0.001 rad` limit activation margin was applied to both couplers, so the result is not source-faithful despite no measured hard-range excursion.
- Release was commanded by opening the gripper. Pad contact count reached zero; during `FREE_SETTLE`, the bottle regained table support. At the end of that 1 s phase, linear velocity was approximately `[-0.00634, -0.01460, 0.000067] m/s` and angular velocity `[0.1252, -0.0541, -0.000070] rad/s`. No stationary final placement is claimed.

### Isolated Simulation-Derived Configuration

The measured pad-midpoint target was placed at `[-0.007, 0, +0.015] m` relative to the compiled bottle-body collision-geom center. The fixture carriage used bias compensation, `kp=30000 N/m`, `kv=350 N s/m`, and a 25 N effort cap. The 30 mm and 50 mm carriage commands were 30.5 mm and 51 mm, respectively, to satisfy bottle-COM lift gates under the fixture dynamics. These are diagnostic setup values, not hardware settings. The bottle itself received no force or state write.

The preserved raw result JSON contains the literal `acceptance_classification` string `NON-ACCEPTANCE while the strict source-limit gate is failed`. That string is stale and conflicts with the same record's `source_joint_limit_status: PASS` and no-violation fields. The correct interpretation is **NON-ACCEPTANCE because the tested model/controller were SIMULATION_ONLY**, not because the measured run crossed the source numeric range. The raw result was not edited. The runner source now emits a neutral classification that keeps physics and source-limit gates separate.

## Bounded X2 Integration Results

The isolated pad midpoint and neutral Robotiq orientation were mapped to the canonical X2 bottle target. The only recorded collision exclusion remained the authorized exact pair `head_pitch_link <-> torso_link`; no additional exclusions were introduced. X2 and Robotiq source geometries and source joint ranges were retained.

| Candidate | Station / approach | Best collision-checked pregrasp result | Gate outcome |
|---|---|---|---|
| `isolated_lift_calibrated_pad_midpoint` | Base `[-0.08, 0.18, 0.68] m`, +X approach | Position error `18.614 mm`; orientation error `0.094920 rad`; source limits valid. This remained outside pose tolerance. The position-only exact-target solution penetrated the table at wrist-roll `-62.909 mm`, base mount `-25.587 mm`, and base `-42.599 mm`. | STATIC_FAIL |
| `isolated_lift_calibrated_plus_y_approach` | Same station, +Y approach (`270 deg`) | Position error `103.974 mm`; orientation error `0.424123 rad`; source limits valid. A clear pregrasp existed, but target alignment failed. The position-only exact-target solution intersected wrist-roll/bottle by `-80.909 mm` and wrist-roll/table by `-31.969 mm`. | STATIC_FAIL |
| `isolated_lift_calibrated_forward_station` | Base `[0.00, 0.16, 0.68] m`, +X approach | Position error `69.564 mm`; orientation error `0.280485 rad`; source limits valid. The best pregrasp was collision-free but missed the pose target. The position-only exact-target solution intersected wrist-roll/table by `-62.934 mm`, base mount/table by `-25.000 mm`, and base/table by `-42.526 mm`. | STATIC_FAIL |

All three candidate result records report `physics_run: false`; all gates for source-limited pregrasp, open-grasp alignment, 30 mm TCP lift, collision-free approach/lift, and static corridor are false. Raw solver iterations, joint poses, collision distances, and pregrasp screenshots are preserved in each candidate's `raw/candidate_trace.json` and `static/pregrasp.png` under the external evidence directory.

These three outcomes are bounded failures only. They do not establish global infeasibility. The next X2 integration attempt needs a newly authorized, evidence-led station/arm-target revision; do not proceed to dynamics from these candidates.

## Integrity and Provenance

- Research branch: `research/issue46-x2-robotiq-m0-20261009`.
- Experiment checkout HEAD: `2a607c7e8bcd01fb93e18a12f1f3209fa7dcfa2c`.
- Runtime: Python `3.10.12`, MuJoCo Python/native `3.3.6`, `MUJOCO_GL=egl`.
- Native MuJoCo library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Menagerie pin: `0059d4335f8156206f63a35662313385f7ad6d74`.
- Robotiq source `robotiq_2f85/2f85.xml` SHA-256: `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.
- X2 source pin: `575cc6b988f976c23550e0db85aa1e5475d3652d`; compiled X2 MJCF SHA-256: `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- Canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Isolated runner SHA-256 used for the 50 mm run: `86332cf625e415f2b71c0f9760970f3956d5cf76b667374f226516565589c8c4`.
- Final compiled isolated fixture SHA-256: `e6df563b3311c998ff3ddb0343b910be5665bcb93b12596c9f034461449edd39`.

External evidence directory: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robotiq-load-transfer-20261009\`. Raw result JSON, contact and physics traces, MP4, PNGs, candidate solver traces, model XML, logs, and checksum manifest are preserved there. The original raw files were checksum-verified before updating this report.

Issue #46 remains open. PR #58 remains Draft and unchanged. No production model or shared contract was modified.

## Reproduction

See [the reproduction instructions](issue46_x2_robotiq_load_transfer_20261009_reproduce.md). The evidence run commands and runtime identities are also recorded in each raw `result.json`; use fresh output directories when rerunning.
