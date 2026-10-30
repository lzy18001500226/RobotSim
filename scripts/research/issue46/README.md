# Issue #46 research tool catalog

This directory is the index for retained Issue #46 research tools. New
maintained code belongs in ROS 2 packages or the MuJoCo scene/model owners,
not beside these probes.

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
