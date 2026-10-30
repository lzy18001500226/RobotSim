# Issue #46 STL visual baseline freeze

Status: `HUMAN_VISUAL_REVIEW=PASS` for the 2026-10-30 full X2-STL plus
RAICOM-room static review.

## Frozen selection

| Item | Identity | Status |
| --- | --- | --- |
| Active gripper model | `X2-OmniPicker-STL` | frozen visual/kinematic selection |
| STL model SHA-256 | `088c1f44963773b6a3cebe0ca3e428b409fa16077bdb930e7b142c5c75981857` | verified by registry loader |
| STL trajectory SHA-256 | `117bc4e81edbef8b341341c1cb81baba5a7bde61b590eaea93553ec77c4b540a` | verified by registry loader |
| Full X2 source for review | `full_x2_stl_static_review.xml` SHA-256 `4a98706f05862190cc12d56fd960656af5795d7938ecedcd22913a2b169acebf` | generated evidence, explicit input only |
| Competition environment | pinned RAICOM room XML SHA-256 `d72618e8d88f1c6417cc239bc3ae690068845665a5db01775ed24dd81edc6c0a` | source environment input |
| Collision proxy manifest | SHA-256 `7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a` | 161 proxy bytes verified before static review |

## Accepted visual composition

- Robot body: the human-approved full X2 STL review fixture, not the separate
  RAICOM runtime X2 body.
- Hands: one 11-body STL assembly per side; visual and collision meshes share
  the same bodies. Collision proxy rendering is hidden by default only.
- Environment: RAICOM room/table/chair attached around the approved robot.
- Object: the approved upright cylinder remains on the attached RAICOM table.

## Explicitly not frozen as qualified behavior

- Native runtime body/controller integration.
- Reset-to-table navigation or randomized-obstacle avoidance.
- RGB-D pose estimation in the live control loop.
- Stable upper-body IK, contact dynamics, grasp, lift, or delivery.

## Tool disposition

| Group | Disposition |
| --- | --- |
| `scripts/research/issue46/grasp/` force/contact scans and historical lift runners | retain as historical reproducibility tools; do not import from runtime |
| `scripts/research/issue46/native_runtime/` and `scene/` mount/native probes | retain as source/runtime diagnostic tools; do not treat as full task runtime |
| `scripts/research/issue46/perception/` | retain as offline perception/docking tools pending ROS adapter implementation |
| Candidate CAD/G2/passive-loop variants | retain as evidence only; not selectable from the active registry |

The next cleanup step is organizational: move retained research tools into
topic directories with compatibility-tested imports, then create maintained
ROS packages for navigation, perception, manipulation, and task execution.
