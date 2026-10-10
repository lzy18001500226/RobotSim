# Issue #46: X2 + Robotiq Mount Replacement Audit

**Outcome:** BLOCKED at the corrected mount/contact gate. The prior 56.295 mm lift remains historical evidence from the previous assembly and is not validation of this model.

## Scope and provenance

The task was limited to stopping transfer work, auditing the right wrist/tool assembly, building a task-specific replacement variant, and producing morphology/contact evidence. No pickup, release, transfer, or placement run was made with the replacement variant.

- RobotSim branch/head at experiment start: `research/issue46-x2-robotiq-m0-20261009` / `72cac19820f2ee6f512da37f532aec50acb27c91`.
- X2 source: `AgibotTech/agibot_x2_urdf`, `575cc6b988f976c23550e0db85aa1e5475d3652d` (Mulan PSL v2).
- MJCF: `X2_URDF-v1.4.0/X2-Ultra.xml`, SHA-256 `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- Tool reference: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Menagerie: `google-deepmind/mujoco_menagerie`, `0059d4335f8156206f63a35662313385f7ad6d74`; Robotiq XML SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.
- MuJoCo Python/native: 3.3.6; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Corrected compiled XML SHA-256: `a7d73a80457a6a8847cc7b6e64f42ab1167a48f40113caeca0128e456f34f7fd`.
- Canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Main model-builder SHA-256: `5954a772be1bede09b7c8f25425b61ddce809516cf316d77ecd5fd11cbbeb063`.
- Morphology runner SHA-256: `14616e2cb4abdfbedf5883a1ed4b54d5cad78ea53c37206bd720416790dafbc7`.

## Assembly audit

The pinned `X2-Ultra.xml` has no separate right-hand body subtree, hand joints, hand actuators, tendons, or equalities. The white hand-like shape in the prior image came from the right-wrist visual mesh; it was not a separate articulated subsystem in this MJCF. The replacement XML changes that visual from `right_wrist_roll_link.stl` to the official tool-variant `right_wrist_roll_extend_link.stl`. It retains the pinned right-wrist collision mesh, body, joint, mass, and inertia. The serialized left-wrist subtree XML matches the source subtree exactly.

Exactly one Robotiq `rq_base_mount` is attached under `right_wrist_roll_link`. Its compiled subtree mass is 1.0526083388427392 kg. The gripper retains its Menagerie articulation, collision meshes, actuators, and coupler structure. Source X2 arm joint ranges are unchanged. The official reference fixed mount is `[0.00196, 0.00035, -0.062] m`, `rpy=[3.14,0,0] rad`; the Robotiq orientation composes this reference with the measured Robotiq insertion/opening axes. This is a reference-derived adapter transform, not an independently verified Robotiq-to-X2 adapter specification.

The original source-faithful Robotiq coupler failure remains recorded. The experimental XML uses the existing 0.001 rad SIMULATION_ONLY coupler-limit activation margin and the exact `head_pitch_link`/`torso_link` diagnostic exclusion. Neither changes the source joint ranges. No vendor assets were edited.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Original right-hand body/joint/actuator subtree absent | PASS structurally | No separate X2 hand bodies, joints, or actuators exist in the pinned MJCF. The tool-variant wrist visual replaces the generic wrist visual. |
| Left hand/wrist preserved | PASS | Source and replacement left-wrist XML subtrees match exactly. |
| Single Robotiq and source arm limits | PASS structurally | One Robotiq root under the right wrist; no arm range delta; XML compiles. |
| Initial unintended robot contacts | PASS | Zero contacts in the reset OPEN state. |
| Mounted OPEN/CLOSE dynamics | FAIL | Pad gap changes 93.4 mm → 21.1 mm → 88.0 mm, but 18 `rq_right_coupler_joint` source upper-limit violations occur. Close also produces wrist/follower contact at -1.102 mm (24.21 N) and wrist/pad contact at -0.122 mm (240.78 N). The pad-to-pad closed stop is separately recorded. Peak arm-joint displacement is `1.607e-5 rad`; no bottle contact or post-start qpos writes occurred. |
| OPEN bottle-alignment geometry | FAIL | IK position residual is 0.106 mm, but the open pose has bottle-body/wrist penetration of 15.360 mm and bottle-shoulder/wrist penetration of 11.458 mm. Minimum pad-to-bottle distance is 0.806 mm. This is not a collision-free approach. |
| Corrected-model physical pickup / release | NOT RUN | The mount and open-clearance gates failed. |
| Transfer/place | NOT RUN | Stopped as instructed. |
| Source-faithful acceptance | NOT CLAIMED | Original coupler-limit failure remains; the replacement uses SIMULATION_ONLY treatment. |

**Decision:** `ROBOTIQ_MOUNT_VALID=FAIL`; overall `MODEL_MORPHOLOGY_PASS=FAIL`. The structural replacement is correct as represented in the pinned MJCF, but the reference-derived Robotiq adapter transform is not mechanically validated and the measured wrist interference rejects it. Do not resume pickup or transfer until the Robotiq-to-X2 adapter transform/interface is resolved and the open alignment is collision-free.

## Evidence

Final diagnostic run: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\mount-replacement-20261010\attempt-03\`.

It contains `result.json`, the compiled replacement XML, raw joint/contact CSV/JSONL traces, six static PNGs, and the no-contact mounted open/close MP4. Earlier attempts are preserved in sibling paths; raw files were not edited after collection.
