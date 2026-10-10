# Issue #46: Dual Robotiq X2 Platform (2026-10-10)

## Outcome

**BLOCKED before gripper motion and bottle manipulation.** The isolated fixed-base X2 model contains two fully namespaced Robotiq 2F-85 instances, preserves the X2 body and arm joint tree, and passes the initial structural and wrist-mount clearance checks. The first MuJoCo physics step violates the pinned Menagerie source upper limit on all four coupler joints by about 13 microradians. Both independent clean-reset gripper trials stop at that same first step. The source limits were not altered and no controller or solver tuning was attempted.

The result is a SIMULATION_ONLY adapter prototype. The official X2 source describes OmniPicker wrist mounts, not a Robotiq mount; composing those frames with the Menagerie insertion/opening axes does not establish hardware compatibility. No bottle-contact or pick-and-place dynamics were run. The previous 56.295 mm diagnostic lift is historical SIMULATION_ONLY evidence and is not evidence for this corrected dual-gripper platform.

## Pinned inputs and identity

- RobotSim branch: `research/issue46-x2-robotiq-m0-20261009`; task changes are based on the existing branch at `1588e3dca37772fc3ad30335259ba6d44485ee89`.
- Loaded RobotSim helper: `scripts/research/issue46_x2_robotiq_m0.py`, SHA-256 `5954a772be1bede09b7c8f25425b61ddce809516cf316d77ecd5fd11cbbeb063`; its committed-HEAD blob is `e6ddac0c1ffe78031d1b14be094bc16c5a1501d5179b1167b830bf3a30c39c06`. The run used the pre-existing dirty worktree copy. Inspection of its diff found no changes to `controller_config` or `apply_controller`, but the exact loaded helper still differs from `HEAD` and is not part of this task's commit. The run's complete porcelain state is captured in `result.json`.
- AgiBot source: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, Mulan PSL v2.
- X2 MJCF: `X2_URDF-v1.4.0/X2-Ultra.xml`, SHA-256 `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- X2 tool-frame reference: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- MuJoCo Menagerie: `google-deepmind/mujoco_menagerie`, commit `0059d4335f8156206f63a35662313385f7ad6d74`, BSD-2-Clause; model `robotiq_2f85/2f85.xml`, SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.
- Runtime: Python 3.10.12, MuJoCo Python/native 3.3.6. Native library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Final runner SHA-256: `6183979f925123f1182b05f81c929bb251ec16db647094948ed2255ba7724e0b`.
- Final generated MJCF SHA-256: `1a8cbf80cf9f186009aba7df17b13f4a89c07ba7b52b567f5a593af57fc7a8c7`.

The complete runtime identity, including canonical asset helper and mesh hashes, is in the external `result.json`.

## Mount and model audit

The pinned OmniPicker URDF provides different fixed wrist references:

| Side | Fixed joint | Parent to child translation (m) | RPY (rad) |
|---|---|---:|---:|
| Left | `L_omnipicker_joint` | `[0.00259, 0.00035, -0.062]` | `[3.14, 0, 3.14]` |
| Right | `R_omnipicker_joint` | `[0.00196, 0.00035, -0.062]` | `[3.14, 0, 0]` |

Each Robotiq root transform was derived independently by composing that side's official reference with the pinned Menagerie jaw insertion/opening frame. These are reference-based simulation adapter transforms only; no Robotiq-specific X2 adapter dimensions or wrist inertia were found in the inspected source. No adapter mass or inertia was added.

The source X2 MJCF has no five-finger hand subtree. All source body names and all source joint names except the intentionally removed `floating_base_joint` are present in the fixed-base build. For every retained source joint, compiled type, axis, limited flag, and range match the source model; both wrist roll joints retain their own motors. Left/right gripper bodies, joints, geoms, meshes, actuators, tendons, equalities, and TCP sites use separate `rq_left_` and `rq_right_` namespaces. Previously anonymous imported objects now receive deterministic side-prefixed names; their geometry and dynamics are unchanged. `result.json` contains each per-side inventory and the namespace diagnostics. Each compiled subtree mass matches the pinned Menagerie model at approximately 1.052608 kg.

The tool URDF pairs the left wrist visual and collision with `left_wrist_roll_extend_link.stl`. On the right, visual uses `right_wrist_roll_extend_link.stl` while collision uses `right_wrist_roll_link.stl`. The generated model follows those mesh assignments. It retains the ordinary X2 MJCF wrist inertias: 0.338 kg left and 0.337 kg right. The inspected tool URDF reports 0.107 kg for each tool wrist link, so this mass/inertia mismatch is unresolved rather than silently substituted.

The accepted table and multipart bottle are used in the static workcell image. For no-contact dynamic preflight only, the bottle is placed far from the robot before the first physics step. That pre-rollout setup is not a manipulation rollout or an active-rollout state write.

## Validation

| Gate | Result | Evidence |
|---|---|---|
| Pinned source and runtime identity | PASS | `result.json` identity block |
| Exactly two independently namespaced grippers | PASS | `result.json` structural inventory |
| X2 body/joint tree preserved; all retained joint types, axes, limit flags and ranges unchanged | PASS | Source/compiled property comparison in `result.json` |
| Both wrist roll joints retained with their motors | PASS | Per-wrist joint/actuator mapping in `result.json` |
| All gripper object names side-prefixed and unique | PASS | Per-side object inventory and namespace diagnostics in `result.json` |
| Unwanted five-finger geometry absent | PASS | Pinned X2 body-name audit |
| Initial source joint ranges | PASS | `initial_source_limit_margins` |
| Initial wrist-to-gripper penetration | PASS | `wrist_mount_penetrations_at_open` is empty |
| No new collision exclusions | PASS | `collision_exclusions_added` is empty |
| Source-faithful coupler limits during dynamics | **FAIL** | First step, 0.001 s; four couplers exceed their 0 rad upper limit |
| Mounted open/close dynamics | **FAIL before closure** | Left-only, right-only, and dual trials stop at step 1 |
| Bottle contact / single-arm pick-and-place | NOT RUN | Stopped at the first mandatory dynamic gate |

At step 1, the observed coupler qpos values were:

| Joint | Source range (rad) | qpos (rad) | Upper-limit overshoot (rad) |
|---|---:|---:|---:|
| `rq_left_right_coupler_joint` | `[-1.57, 0]` | `1.2785894667e-5` | `1.2785894667e-5` |
| `rq_left_left_coupler_joint` | `[-1.57, 0]` | `1.3597595708e-5` | `1.3597595708e-5` |
| `rq_right_right_coupler_joint` | `[-1.57, 0]` | `1.2778420418e-5` | `1.2778420418e-5` |
| `rq_right_left_coupler_joint` | `[-1.57, 0]` | `1.3629613735e-5` | `1.3629613735e-5` |

The right-only and left-only trials both fail this same all-model source-limit audit on their first step, before their actuator command can move the jaw. The isolated traces contain no contacts, and the runner records zero qpos writes after rollout start. The previous model's larger source coupler excursions remain preserved in earlier evidence; this run did not change limits, gains, effort caps, or solver settings.

The source X2 neutral head/torso audit also records an existing source contact distance of `-0.00711246 m`. It was retained without adding an exclusion. The structural gate's wrist/gripper clearance check passes; this head/torso source behavior is reported separately.

## Visual evidence

External evidence directory:

`C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\dual-robotiq-platform-20261010\`

- Four table-and-bottle-hidden full-body views: front, side, rear, and three-quarter. The table and bottle remain in the physical model and are shown together in the workcell view.
- Left/right table-and-bottle-hidden open-jaw wrist close-ups.
- Left/right kinematic closed-pose renders. These are static visualization states only.
- Wrist collision overlay and canonical table/bottle workcell overview.
- `dual_gripper_static_pose_review.mp4` is a labeled static pose comparison, **not physics evidence**.
- `dual_gripper_dynamic_until_first_failure.mp4` shows the dynamic run only through its first failed step.
- Raw combined and independent traces are in `raw/`.

An older `dual_gripper_open_close.mp4` from an earlier runner iteration is retained in the evidence directory as historical output; it is not the current run's completed open/close result. Use the current result paths above.

## Conclusion

The dual-gripper morphology and model namespacing are implemented and structurally validated as a SIMULATION_ONLY prototype. The source property audit verifies every retained X2 joint, and the namespace audit verifies every instantiated gripper object including previously anonymous geoms. The platform does not pass source-faithful dynamics because the Menagerie coupler coordinates leave their unchanged source limits at the first step. Therefore no mounted close/reopen, bottle hold, lift, transfer, or release was accepted or run. The next required engineering decision is how to represent the source Robotiq coupler dynamics without weakening source limits; this report does not apply a correction.
