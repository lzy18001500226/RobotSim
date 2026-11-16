# Issue #46 research tool catalog

This directory is the index for retained Issue #46 research tools. New
maintained code belongs in ROS 2 packages or the MuJoCo scene/model owners,
not beside these probes.

## Current common task entry

Use `grasp/table_edge_grasp_trials.py` for the 22 cm bottle task. Select a
position with `--case 0` through `--case 10`; do not create a copy of the
program for each position. The current complete-transfer profile is:

```bash
python -B -u scripts/research/issue46/grasp/table_edge_grasp_trials.py \
  --output-dir "$X2_REVIEW_ROOT" --case 5 --visible --far-observe \
  --fast-task-timing --complete-transfer --carry-baseline-78df5bd
```

Set `X2_REVIEW_ROOT` to the canonical, governed experiment output directory
described in [evidence path governance](../../../docs/setup/evidence-path-governance.md).
Use the MuJoCo virtual environment and the existing X2 task worktree. Exact
measured revisions, reproduction commands and acceptance limits are in the
[handoff](../../../docs/workflows/X2_TRANSFER_HANDOFF.md).

The public entry delegates to these existing owners:

| Component | Responsibility | Disposition |
| --- | --- | --- |
| `grasp/table_edge_grasp_trials.py` | Case configuration, common entry and experiment result | Retain active research prototype |
| `grasp/live_retreat_turn_review.py` | Full episode orchestration and review launch | Retain active research prototype |
| `native_runtime/xvfb_native_reset_stand.py` | Native simulator/MC lifecycle and physical audits | Retain active research prototype |
| `native_runtime/native_task_grasp_region.py` | Observation, approach and feasible grasp candidates | Retain active research prototype |
| `native_runtime/standing_waist_preposition.py` and `native_grasp_stand_restore.py` | Forearm preparation, posture and owner handoff | Retain active research prototype |
| `native_runtime/carry_baseline_78df5bd.py` | Hash-checked transport strategy from the immutable Git commit | Retain active compatibility adapter |

The compatibility adapter reads Git objects from `78df5bd`; that commit is an
active input, not disposable history. It restores transport strategy while
keeping current I/O guards and the current low-release placement policy.

The 0/5/10 physical results used successive revisions `31009ed`, `8fd5257`
and `9febb26`, respectively. `15d7072` records those results. They do not
establish a frozen-final-revision 3/3 runner batch, the other eight complete
transfers, LiDAR localization or hardware deployment.

## Retention boundaries

The contact/fit viewers below are retained demonstration and research tools.
The 2026-11-16 hygiene inventory found no byte-identical Python, shell or HTML
source copies in this directory; similar filenames alone do not justify
deletion. Existing local modifications and untracked research tools retain
their exact contents and are inventoried separately from current task changes.

Keep active prototypes in place until a separately validated runtime migration
replaces them (`PORT-TO-C++`); retain offline viewers and diagnostic tools as
`KEEP-AS-TOOL`. This catalog does not promote them into robot runtime or start
a C++ migration. Permanent regression tests remain maintained coverage.

Historical evidence archival is described in
[the archival procedure](../../../docs/setup/evidence-path-governance.md#historical-trace-archives).
It does not change source, models, recorded verdicts or current review assets.

## Scene and model review

- `scene/full_x2_stl_review.py`: bilateral STL human-review viewer.
- `scene/raicom_autonomy_scene_review.py`: explicit-input composition of an
  approved full-X2 review fixture and the RAICOM room.
- `scene/raicom_stl_mount_audit.py`: source wrist-mount audit.
- `scene/raicom_stl_mount_review.py`: mount-review viewer.

## Native-runtime diagnostics

- `native_runtime/stl_hand_overlay.py`
- `native_runtime/stl_hand_startup.py`
- `native_runtime/poststop_{state_audit,hold,review}.py`

These diagnose a pinned simulator/MC delivery. They are not the eventual arm,
navigation, or task runtime.

## Offline perception and docking

- `perception/rgbd_truth_capture.py`, `perception/rgbd_grasp_contract.py`
- `perception/foundationpose_input.py`, `perception/foundationpose_compare.py`
- `perception/upright_cylinder_refine.py`
- `perception/table_approach_{static_ik,review}.py`

Ground truth is permitted only as an offline estimator comparator. A maintained
perception path must publish timestamped camera-frame estimates through ROS 2.

## Historical contact/grasp exploration

The `grasp/` static scans, force maps, path generators and lift runners
preserve reproducibility of past evidence. They must not be used as implicit
task runtime, model selection, or source of a mechanical claim.

## Migration rule

Before moving a retained tool into a topic subdirectory, update every dynamic
import, command, test, and evidence reference in one compatibility-tested
change. Do not use path-only moves as a substitute for an interface contract.
