# Issue #46 X2 + Robotiq Physical Grasp Recovery

**Outcome:** `BLOCKED — the Robotiq coupler violates its unchanged source limit on the first physics step, the isolated bottle stays table-supported, and no tested X2 pregrasp meets the source-limited collision and pose gates.` This bounded result does not establish global X2 + Robotiq infeasibility.

## Provenance

- Research branch: `research/issue46-x2-robotiq-m0-20261009`
- Experiment checkout HEAD: `366c8f716b2cf25042c2ab653a4a39939919ec39`; the research scripts were working-tree changes at execution. Publication commit is recorded in the GitHub closeout.
- X2: `AgibotTech/agibot_x2_urdf`, `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Menagerie: `google-deepmind/mujoco_menagerie`, `0059d4335f8156206f63a35662313385f7ad6d74`; Robotiq `robotiq_2f85/2f85.xml`, BSD-2-Clause.
- Runtime: Python `3.10.12`, MuJoCo Python/native `3.3.6`, `MUJOCO_GL=egl`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- X2 `X2-Ultra.xml` SHA-256: `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`; OmniPicker URDF SHA-256: `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Menagerie `2f85.xml` SHA-256: `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`; canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Canonical bottle and table definitions were unchanged. Bottle: 70 mm diameter, 244.5 mm overall height, 0.570 kg.
- The only X2 collision exception is the existing `SIMULATION_ONLY` exclusion `head_pitch_link <-> torso_link`; original collision geometry remains present. No vendor checkout, production model, source joint limit, actuator cap, bottle, or table was edited. PR #58 remains a separate Draft and was not changed; Issue #46 remains open.

Raw machine-readable data, complete traces, renders and videos are in:
`C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robotiq-physical-recovery-20261009\`.

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| `COUPLER_SOURCE_LIMIT` | **FAIL** | Both source coupler ranges are `[-1.57, 0] rad`, and the reset state is exactly at the upper stop. On the first 1 ms physics step, constraint force drives each qpos positive (`+1.2093e-5 rad`) with about `+0.0129 Nm` generalized constraint force and `+12.6 rad/s^2` acceleration. No actuator force is applied on that step. The maximum excursion in the isolated run is `+0.0005454 rad` right and `+0.0004891 rad` left. The tested 0.5 ms, gravity-off, interior-target and constraint-consistent initialization probes did not establish strict source-limit compliance. The `-0.001 rad` initialization previously worsened the excursion and was not repeated. No range or equality was relaxed. |
| `ISOLATED_BILATERAL_CONTACT` | **PASS, diagnostic only** | With the one fixture-only `-12 mm X` placement correction, close and 1 s hold produced real contact on both pads. Hold persistence was `1000/1000` samples; summed pad normal forces were about `50.1–52.7 N` per side. No non-gripping robot-to-bottle contact occurred in this corrected run. Because the coupler source-limit gate fails, the dynamics remain `NON-ACCEPTANCE`. |
| `ISOLATED_PHYSICAL_LIFT_50MM` | **FAIL** | A 1 mm carriage command moved bottle COM only `0.167 mm`; maximum measured COM rise was `0.224 mm`. The bottle remained in table contact (table normal force about `7.0–9.0 N` during the 1 s hold). The 1 mm milestone failed its 80% progress gate, so 5 mm, 30 mm and 50 mm milestones were not attempted. No load transfer or airborne lift was demonstrated. |
| `X2_COLLISION_FREE_PREGRASP` | **FAIL / NOT DEMONSTRATED** | Three source-limited station/approach/grasp-height configurations and deterministic in-range arm seeds were evaluated. None met the TCP error gates (`3 mm`, `0.02 rad`) together with collision clearance. This is not a global reachability result. |
| `X2_DYNAMIC_APPROACH` | **NOT RUN** | Blocked by the pregrasp gate. |
| `X2_PHYSICAL_LIFT_50MM` | **NOT RUN** | No accepted X2 approach or grasp existed. |
| `TRANSFER_RELEASE_SETTLE` | **NOT RUN on X2** | The isolated diagnostic fixture opened and released the bottle, but did not achieve load transfer or a free lift; this is not an X2 transfer/place result. |

### Stage 1: Coupler Diagnosis

At reset the coupler equalities are position-consistent, but each hinge begins at its source upper limit. Constraint impulse, rather than an out-of-range actuator target or bottle contact, causes the first-step violation. Halving the timestep and disabling gravity do not remove it. An interior driver target derived from the Menagerie actuator map (`0.01 rad`, control `3.1874999004`) permits a clean constraint-consistent initial configuration and passes a 10 ms no-contact probe. The unchanged 1.9 s OPEN/CLOSE/REOPEN cycle later crosses the source upper limit at step 27 and peaks at `+0.00087199 rad` right / `+0.00087119 rad` left at step 31. Its pad gaps are `92.26 / 14.58 / 85.38 mm` (open/closed/reopened), so motion is functional while source-limit compliance still fails. The evidence supports a numerical closed-loop constraint/endpoint interaction in the compiled model; it does not prove that the source hardware is incompatible. No physically defensible correction preserving the source range and required loop constraints was established.

The earlier mounted gripper OPEN/CLOSE/REOPEN motion result remains a functional motion observation only. It was not rerun as part of this recovery and did not pass the strict coupler source-limit gate.

### Stage 2: Isolated Bottle Fixture

In the unshifted attempt, the open fixture had only `5.67 mm` minimum non-gripping clearance. During CLOSE, `bottle_body` contacted Robotiq `geom_4` at step 856 (`0.856 s`), with `23.08 N` normal force, peaking at approximately `26.26 N`; the bottle COM moved backward by about `6.2 mm`. This justified a single `12 mm` mount offset away from the measured contact surface. The corrected open preflight had `13.63 mm` minimum non-gripping clearance and no initial contact or source-limit violation.

The corrected dynamic run removed that base/bottle collision and retained bilateral pad contacts. However, the table continued to support the bottle, the carriage's 1 mm motion did not produce a meaningful pickup, and the couplers again violated their hard limits at step 1. This is not a valid supported grasp or lift. The free bottle was never welded, attached, teleported, mocap-driven or qpos-written after rollout start; the opposite hand was not present in the isolated fixture.

### Stage 3: X2 IK and Collision Audit

The revised solver evaluates signed robot/table and robot/bottle geom-pair distances during each full-pose least-squares evaluation and in interpolated path checks. It records all candidate iterations. When native `mj_geomDistance` returns zero with a non-degenerate witness, the solver cross-checks the legacy distance path and the compiled collision pipeline with a temporary query margin that is restored immediately. This is a diagnostic query; source collision properties are not changed. The native-zero/witness discrepancy was locally reproduced for a separated convex mesh pair; see [MuJoCo issue #3383](https://github.com/google-deepmind/mujoco/issues/3383).

| Configuration | Position-only FK | Collision-aware selected pregrasp | Finding |
|---|---:|---:|---|
| Right-side, body center, 90-degree approach | `~4.1e-14 m` target error; the exact-position seed penetrates the tabletop by `65.6 mm` at right wrist-roll and `21.6 mm` at the mount | `53.3 mm`, `0.0812 rad` | Clearance can be found only after losing the target pose; fail. |
| Natural-front station, body center, 0-degree approach | `~3.0e-15 m`; exact-position seed has `57.7 mm` base/table and `54.5 mm` wrist/table penetration | `64.2 mm`, `0.2969 rad` | Fail pose and corridor gates. |
| Right-side, 25 mm lower body target, 90-degree approach | `~2.2e-13 m`; exact-position seed penetrates the tabletop | `68.1 mm`, `0.1074 rad` | Fail pose gate; closest selected table clearance is only `0.22 mm`. |

The pose errors are far beyond `3 mm / 0.02 rad`; no grasp, lift, or interpolated approach corridor was accepted. Exact joint poses, source ranges, all per-evaluation distance records and screenshots are in the candidate traces. The bounded failures identify both real tabletop collision for exact-position seeds and an unresolved IK/station pose gap. They do not prove global geometric infeasibility.

## Failure Classification

- **Source-model incompatibility:** not established. Neither the coupler nor the gripper/bottle evidence proves the source hardware cannot perform the task.
- **Numerical constraint issue:** established for this compiled Robotiq model. The closed-loop constraint drives the coupler coordinates outside their source upper stops from a reset state, before bottle contact.
- **Fixture geometry:** the original centered fixture allowed a real bottle/base collision during close; one measured 12 mm placement correction removed that contact in static and dynamic evidence.
- **Physical load transfer:** failed in the corrected isolated run. Bilateral pad force persisted, but the bottle remained table-supported and did not follow the carriage meaningfully. The recorded evidence does not isolate whether pad friction, contact geometry, or carriage motion is the limiting factor; no controller or friction tuning was attempted.
- **X2 geometry / IK:** exact-position seeds had real tabletop penetration, while collision-aware solutions missed the target pose by 53–68 mm. This is a bounded reachability/IK failure, not proof of global infeasibility.

## Reproduction

See [`issue46_x2_robotiq_physical_recovery_20261009_reproduce.md`](issue46_x2_robotiq_physical_recovery_20261009_reproduce.md) for exact commands. The highest-value reproductions are the coupler attempt 03, corrected isolated bottle attempt 03, and signed-distance IK attempt 04.
