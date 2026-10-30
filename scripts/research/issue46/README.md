# Issue #46 research tool catalog

This directory is the index for retained Issue #46 research tools. The current
flat `scripts/research/issue46_*.py` names are historical compatibility paths;
they are not runtime imports. New maintained code belongs in ROS 2 packages or
the MuJoCo scene/model owners, not beside these probes.

## Scene and model review

- `issue46_stl_full_x2_review.py`: bilateral STL human-review viewer.
- `issue46_x2_stl_raicom_autonomy_scene.py`: explicit-input composition of an
  approved full-X2 review fixture and the RAICOM room.
- `issue46_raicom_stl_mount_audit.py`: source wrist-mount audit.
- `issue46_raicom_stl_mount_review.py`: mount-review viewer.

## Native-runtime diagnostics

- `issue46_raicom_stl_native_hand_overlay.py`
- `issue46_raicom_stl_native_hand_startup.py`
- `issue46_raicom_stl_poststop_{state_audit,hold,review}.py`

These diagnose a pinned simulator/MC delivery. They are not the eventual arm,
navigation, or task runtime.

## Offline perception and docking

- `issue46_rgbd_truth_capture.py`, `issue46_rgbd_grasp_contract.py`
- `issue46_foundationpose_input.py`, `issue46_foundationpose_compare.py`
- `issue46_upright_cylinder_refine.py`
- `issue46_stl_table_approach_{static_ik,review}.py`

Ground truth is permitted only as an offline estimator comparator. A maintained
perception path must publish timestamped camera-frame estimates through ROS 2.

## Historical contact/grasp exploration

The remaining `issue46_stl_*` static scans, force maps, path generators and
lift runners preserve reproducibility of past evidence. They must not be used
as implicit task runtime, model selection, or source of a mechanical claim.

## Migration rule

Before moving a retained tool into a topic subdirectory, update every dynamic
import, command, test, and evidence reference in one compatibility-tested
change. Do not use path-only moves as a substitute for an interface contract.
