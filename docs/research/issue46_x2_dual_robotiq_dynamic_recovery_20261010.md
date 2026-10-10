# Issue #46: Dual Robotiq Dynamic Recovery (2026-10-10)

## Result

**BLOCKED before bottle dynamics.** The pinned Robotiq source has a reproducible first-step coupler violation caused by its spring preload at the zero upper stop. The authorized `0.001 rad` simulation-only activation margin removes that first-step violation without changing source ranges. With that diagnostic treatment and the previously authorized exact `head_pitch_link`/`torso_link` exclusion, left OPEN/CLOSE/OPEN passes, but right and both-hand cycles stop at a wrist/follower collision. A separate static check of the current canonical workcell reproduces wrist/bottle penetration on the approach path. No bottle physics was run in this task.

The earlier uncommitted report [issue46_x2_robotiq_complete_demo_20261010.md](issue46_x2_robotiq_complete_demo_20261010.md) claims a prior SIMULATION_ONLY 50 mm lift and complete pick/place. Its referenced raw evidence directory was absent when checked. The claim and report are preserved unchanged, but that episode is not independently verified here and was not rerun.

## Provenance

- RobotSim branch/head: `research/issue46-x2-robotiq-m0-20261009` / `ce9d3bcf99cc0b895cfee6b11b7dc2c6d3c6c84d`.
- X2 source: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- X2 MJCF `X2_URDF-v1.4.0/X2-Ultra.xml`, SHA-256 `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- X2 tool-variant URDF `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- MuJoCo Menagerie commit `0059d4335f8156206f63a35662313385f7ad6d74`; `robotiq_2f85/2f85.xml` SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.
- Runtime: Python `3.10.12`, MuJoCo Python/native `3.3.6`, `MUJOCO_GL=egl`. Native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Controller helper snapshot: `5954a772be1bede09b7c8f25425b61ddce809516cf316d77ecd5fd11cbbeb063`; it is a byte-identical snapshot of the uncommitted helper used by the trial. No controller/helper source was changed by this task.
- Canonical manipulation helper: SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Coupler diagnostic/static runner currently preserved at `scripts/research/issue46_x2_robotiq_coupler_recovery.py`, SHA-256 `2d9effdb3aa69a00dc1e4c1888764fb8c424ec78cbe7f1d23e4843638f88d752`. The full diagnostic result records this same hash. The earlier no-exclusion result records runner hash `6ac4fa758e82991d247233e791ca704df9fd4bdbf9db5631ac4db04802c0cf0a`; that earlier runner source snapshot was not retained. Its raw result, traces, videos, and compiled XMLs are retained.

## Stage 1: Coupler Start-Up

The pinned Menagerie model reproduces the error in isolation. The two spring-link joints have source reference `2.62 rad`, stiffness `0.05 N m/rad`, and produce `+0.131 N m` passive preload at reset. With the default/open command, a single `1 ms` step moves both couplers from `0` to `+44.4454 microrad`, beyond the source range upper stop of `0 rad`. An explicit OPEN command gives the same result, so controller startup is not the cause. A close command instead moves them to `-0.718119 mrad`, inside the source range.

In the dual X2 assembly without the controller, the four first-step source-limit exceedances are `12.8–13.3 microrad`; with the controller they are `12.8–13.6 microrad`. The effect exists in the isolated source model and is slightly smaller in the dual assembly. This is source-model spring preload/stop interaction, not a dual-mounting defect or an OPEN-command initialization error.

## Stage 2: Source-Faithful vs Diagnostic

The SOURCE_FAITHFUL model remains **FAIL**: all four coupler ranges stay `[-1.57, 0] rad`, and all four exceed `0 rad` at the first step. This verdict is preserved independently.

The SIMULATION_ONLY diagnostic assigns `jnt_margin=0.001 rad` to the four couplers. It does not widen `jnt_range`, alter the spring, friction, or actuator caps, or disable limits. The first step moves the couplers to approximately `-38.3 microrad`; the four source ranges remain unchanged and the recorded source-limit violation count is zero during the diagnostic cycles.

The original X2 model also has a reset overlap between `head_pitch_link` and `torso_link` of `7.112 mm`. Without an exclusion, the no-object cycles stop at `0.405 s` when `head_pitch_joint` exceeds its source upper limit by `1.003 mrad`. The diagnostic run uses only the previously authorized exact-pair exclusion. This remains a known simulation-only exception, not a source-faithful pass.

## Stage 3: Actuator-Driven Gripper Cycles

All cycles were actual MuJoCo physics with the pinned Menagerie actuators. There was no bottle, no active-rollout qpos write, no coupler source-limit violation in the diagnostic cycles, and all recorded state remained finite. Peak hand-system joint speed was `1.890 rad/s`.

| Mode | Result | Pad gap | First failure / contacts |
|---|---|---|---|
| Left only | PASS, 1,700 steps / 1.700 s | `93.400 mm` open; `8.874 mm` minimum; `91.262 mm` after reopen | 397 expected opposing pad-to-pad contact samples while empty; peak normal force `6.675 N`; no wrist/gripper contact |
| Right only | FAIL at step 584 / `0.584 s` | `93.400 mm` open; `43.915 mm` at stop | `right_wrist_roll_link` vs `rq_right_right_follower`; `0.0853 mm` penetration, `15.369 N` normal force |
| Both | FAIL at step 584 / `0.584 s` | left `43.908 mm`, right `43.915 mm` at stop | Same right wrist/follower collision; `0.0853 mm` penetration, `15.369 N` normal force |

Right-only and both-hand failure occur before closure. They are not bottle-contact failures. The current dual-gripper dynamic gate therefore does not pass.

## Stage 4: Canonical Workcell Corridor

The static check reused the canonical multipart bottle (`70 mm`, `244.5 mm`, `0.570 kg`), canonical G1 table placement, the accepted fixed X2 station, and the byte-pinned helper above. It checked 101 states from OPEN pregrasp to grasp with `mj_forward` only; it did not call `mj_step`. Bottle qpos remained constant and there were zero active-rollout bottle qpos writes.

The source-limited grasp and pregrasp IK position errors were `0.106 mm` and `0.080 mm`, but the configuration/path fails collision clearance:

- At the first pregrasp sample, `bottle_body` penetrates `right_wrist_roll_link` by `1.371 mm`.
- Worst wrist/bottle distance along the path is `-15.360 mm`, reproducing the prior reported collision; the grasp endpoint also intersects `bottle_shoulder` by `11.458 mm`.
- The non-pad gripper-to-bottle minimum is positive `4.914 mm`.
- Wrist-to-table minimum is positive `88.613 mm`. The broader arm-to-table query returned `0.0 mm` at an elbow geom, and the gripper-to-table query returned `0.0 mm` at a pad geom. No active table contact was present in the static contact records; treat these zero-distance queries as unresolved/tangential, not as proof of positive whole-arm clearance.

Because the wrist already intersects the bottle before any physics step, the physical bottle sequence was not run. This reproduces the old `15.360 mm` finding and blocks M0 continuation at the corridor gate.

## Wrist Variant Boundary

The assembled adapter retains the standard X2 Ultra MJCF `right_wrist_roll_link` inertia: mass `0.337 kg`, COM position `[0.003263, 0.001281, -0.086468] m`, and diagonal inertia `[0.00089008, 0.000748845, 0.000240465] kg m^2` in the MJCF inertial frame. The pinned OmniPicker URDF variant gives the same-named link mass `0.107 kg`, inertial origin `[-0.00165168, 0.00009437, -0.03438670] m`, and tensor entries `ixx=0.00007557`, `ixy=0.00000016`, `ixz=0.00000424`, `iyy=0.00014016`, `iyz=0.00000025`, `izz=0.00008717 kg m^2`.

The adapter also uses the official tool mount transform `[0.00196, 0.00035, -0.062] m`, RPY `[3.14, 0, 0]`, while its wrist collision remains the standard MJCF geometry and the tool-variant wrist visual is `right_wrist_roll_extend_link.stl`. This is a SIMULATION_ONLY mixed-variant adapter. These wrist inertias are not interchangeable and were not silently merged; no inertia retuning was attempted here.

## Evidence and Reproduction

Raw outputs, exact XMLs, controller snapshot, traces, screenshots, and actuator-driven MP4s are in the external packet:

`C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-dual-robotiq-recovery-20261010\`

See `REPRODUCE.md` in that packet for commands. `SHA256SUMS.txt` inventories the copied files. The report-only previous 50 mm claim is retained separately from verified raw evidence. Issue #46 remains open; PR #58 was not modified.
