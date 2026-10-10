# Issue #46: Dual-Robotiq Right-Mount Recovery (2026-10-10)

## Result

**BLOCKED — no right-side adapter candidate passed the mounted collision gate.** The source-mount right cycle contacts the right wrist-roll collision mesh with the right follower at step 584. The only authorized alternate tested was a 180-degree adapter-local-Y half-turn at the same mount origin. It completes the actuator trajectory but creates persistent pad/elbow and spring-link/wrist-yaw contacts; right jaw travel stalls at 88.3 mm from a 93.4 mm open gap. The left control remains functional. No bottle physics was eligible or run.

This report supplements [the prior dual-gripper recovery report](issue46_x2_dual_robotiq_dynamic_recovery_20261010.md). It does not repeat or revise the coupler-startup finding. SOURCE_FAITHFUL remains FAIL; the previously authorized 0.001 rad SIMULATION_ONLY coupler activation margin and exact head/torso exclusion are used only for diagnostic dynamics.

## Source and Mount Audit

Pinned sources are AgiBot `agibot_x2_urdf` commit `575cc6b988f976c23550e0db85aa1e5475d3652d` and MuJoCo Menagerie commit `0059d4335f8156206f63a35662313385f7ad6d74`. The tool URDF SHA-256 is `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`; Menagerie `2f85.xml` SHA-256 is `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.

| Side | Pinned tool joint | Parent to child | XYZ (m) | RPY (rad) | Wrist collision mesh |
|---|---|---|---|---|---|
| Left | `L_omnipicker_joint` | `left_wrist_roll_link` → `L_omnipicker_base_link` | `[0.00259, 0.00035, -0.062]` | `[3.14, 0, 3.14]` | `left_wrist_roll_extend_link.stl` |
| Right | `R_omnipicker_joint` | `right_wrist_roll_link` → `R_omnipicker_base_link` | `[0.00196, 0.00035, -0.062]` | `[3.14, 0, 0]` | `right_wrist_roll_link.stl` |

The right visual still references `right_wrist_roll_extend_link.stl`; its collision references the standard `right_wrist_roll_link.stl`. The left visual and collision both use the extended mesh. Hashes: left extended `f379d912d341e6224ec16a62ec9a3c21621e43b6aae1bec0ba93e9a947d7507f`, right extended visual `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`, right standard collision `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`.

The tested source-mount adapter uses the fixed transform values above; no left/right transform transcription error was found. In the compiled gripper basis, local Z is the insertion axis and local −X is the jaw-opening axis. This is a Robotiq-on-X2 SIMULATION_ONLY adapter: the pinned fixed joints define OmniPicker mounts, not a vendor-qualified Robotiq bracket. The collision result therefore identifies a modeled assembly-envelope conflict, not physical-hardware incompatibility.

## Candidate A: Pinned Mount

The preserved original diagnostic assembly had no active right wrist/gripper contact at reset. On right closure, the first contact was step 583 (`t=0.583 s`); at step 584 (`t=0.584 s`) `right_wrist_roll_link` contacted `rq_right_right_follower`, with `0.085261 mm` penetration and `15.369 N` normal force. The right pad gap was `43.915 mm` at stop. The left-only cycle completed 1,700 steps without an external gripper/robot contact and reached `8.874 mm` minimum pad gap.

The distinct left/right transforms and wrist collision meshes are source facts. The evidence points to their interaction with the added Robotiq sweep envelope: the source-mount right follower enters the right wrist collision surface during closure, while the left cycle does not. This does not establish that the original X2 OmniPicker itself has a collision defect.

## Candidate B: Local-Y Half-Turn

At unchanged translation `[0.00196, 0.00035, -0.062] m`, candidate B applies `R_new = R_source * Rotation.from_euler('y', pi)`. Local Y is orthogonal to both insertion and jaw opening; this half-turn reverses both local Z insertion and local −X opening. It is an experimental orientation flip, not a roll about insertion and not a verified production correction. The XML and model checks confirm only the right adapter quaternion changes; joint ranges, actuator force ranges, collision masks, body masses, and body inertias are identical.

The narrow cycle runner returned PASS because its legacy stop guard only watched `right_wrist_roll_link`. The full trace audit found 5,870 right-gripper-to-robot contact samples in each of the right-only and simultaneous cycles:

| Contact pair | First step / time | Worst penetration | Peak normal force |
|---|---:|---:|---:|
| `right_elbow_link` – `rq_right_right_pad` | 277 / 0.277 s | 0.028606 mm | 31.960 N |
| `right_wrist_yaw_link` – `rq_right_right_spring_link` | 292 / 0.292 s | 0.160783 mm | 9.688 N |
| `right_wrist_yaw_link` – `rq_right_right_pad` | 316 / 0.316 s | 0.014897 mm | 34.660 N |
| `right_wrist_yaw_link` – `rq_right_left_spring_link` | 318 / 0.318 s | 0.459102 mm | 19.600 N |
| `right_elbow_link` – `rq_right_left_pad` | 320 / 0.320 s | 0.091141 mm | 36.691 N |

The actuator-driven trajectory ran 1,700 steps and remained finite, with no coupler limit violations under the diagnostic margin and no rollout qpos writes. But the right gap only closed from `93.400 mm` to `88.298 mm`, then reopened to `92.533 mm`; the actuator reached its `5.0` model force cap. That is not a successful gripper cycle. In the simultaneous run the left reached `8.874 mm`, while the right again stalled at `88.298 mm`. Candidate B's left-only cycle had no external robot/gripper contacts. The comprehensive right and simultaneous collision gates are FAIL.

The initial `mj_geomDistance` inventory includes overlapping parent/child assembly meshes that MuJoCo filters from active contacts; those raw distances are retained for inspection but are not treated as collision verdicts. The active-contact trace is the gate evidence.

### Contact-Normal Replay

To recover contact-frame normals missing from the original traces, a deterministic replay stopped at the already recorded first-contact samples only: candidate A steps 583–584 and candidate B steps 277, 292, 316, 318, and 320. Each replayed contact matched its saved geometry pair, distance within 1 micrometer, and position within 10 micrometers; measured position deltas were exactly zero. The world-frame normals below come directly from `mjContact.frame[:3]`:

| Candidate / step | Contact pair | World contact normal |
|---|---|---|
| A / 584 | `right_wrist_roll_link` – `rq_right_right_follower` | `[-0.029113241, 0.807202119, -0.589556748]` |
| B / 277 | `right_elbow_link` – `rq_right_right_pad` | `[-0.001139049, 0.988397352, -0.151886067]` |
| B / 292 | `right_wrist_yaw_link` – `rq_right_right_spring_link` | `[0.012427571, -0.895532388, -0.444822770]` |
| B / 316 | `right_wrist_yaw_link` – `rq_right_right_pad` | `[-0.000010031, -0.978125221, 0.208016949]` |
| B / 318 | `right_wrist_yaw_link` – `rq_right_left_spring_link` | `[0.012009346, 0.657754632, -0.753136521]` |
| B / 320 | `right_elbow_link` – `rq_right_left_pad` | `[0.000041623, -0.978651464, 0.205526909]` |

The complete replay rows, including both A samples, contact forces and exact comparisons, are in `contact-audit/contact_normal_replay.json`. This is a trace-validation replay, not a new full-cycle result.

## Grasp Corridor and Physical Manipulation

Stage 3/4 corridor work was stopped after the alternate adapter failed the mounted no-object collision gate. The previously preserved source-mount workcell check remains a static FAIL: a 101-sample path had `-1.371 mm` wrist/bottle penetration at pregrasp and `-15.360 mm` worst wrist/bottle distance; its grasp endpoint also intersected the bottle shoulder. This task did not rerun that path or alter the bottle/table scene. A revised candidate corridor is **NOT RUN**. Bottle contact, lift, transfer, release, and placement are **NOT RUN**.

No collision geometry, filters, source limits, bottle properties, friction, effort caps, or controller gains were changed. No weld, object teleport, hidden support, or active-rollout bottle qpos write was used.

## Evidence and Reproduction

Durable evidence is in `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-dual-robotiq-recovery-20261010\mount-corridor-recovery-20261010\`. It includes the frozen prior failure, candidate XML, three actuator-driven videos and traces, full robot/gripper contact audit, and before/after collision frames. `TRANSFORM_LABEL_CORRECTION.md` explains the original run artifact's inaccurate word “roll”; the raw result and exact executed script are preserved unchanged. `REPRODUCE.md` records commands, runtime, model identity, and script hashes.

The X2 and Menagerie source checkouts, model geometry, joint limits, and gripper actuation were not changed. Issue #46 remains open and PR #58 was not modified. The next required dependency is an engineering-approved Robotiq bracket orientation/offset or a vendor-qualified right-side mount envelope; no third geometry candidate is attempted in this bounded task.
