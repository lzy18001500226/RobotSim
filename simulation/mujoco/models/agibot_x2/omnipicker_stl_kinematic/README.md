# X2-OmniPicker-STL

Classification: `KINEMATIC_SIMULATION_ONLY`.

This package is the selected **review candidate** for the STL-based X2
OmniPicker. It preserves the original manufacturer STL meshes and the two
official root ranges. It uses the exact 101 source-STL poses that passed human
visual review at commands 0–100, with linear interpolation only between
adjacent samples.

## Explicit model boundary

- `R_hand_narrow2_Link` and `R_hand_wide2_Link` are fixed.
- The root, wide-root mimic, narrow3, wide3, narrow loop, and wide loop are
  driven together from `trajectory.csv`.
- The driver writes the six recorded joint positions and velocities, then
  calls `mj_forward`. It does not use `mj_step` to discover passive motion.
- Seven documented internal assembly collision pairs are excluded. The eight
  source-STL external collision geoms remain enabled.
- This is not a manufacturer-equivalent transmission or a physical grasp
  model. External-contact, load-transfer, and bottle-grasp claims are
  unqualified.

The 101-row table is source-controlled in this package so runtime selection
never reads an ad-hoc `reviews/` path. Its SHA-256 is locked in
`../registry.json`.

## Selection

Use the central registry/loader rather than loading XML by path:

```python
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (
    apply_stl_kinematic_pose,
    load_registered_trajectory,
)

loaded = load_x2_omnipicker("X2-OmniPicker-STL")
trajectory = load_registered_trajectory(loaded.package_path)
# command_percent is within 0..100.
```

The registry's proposed default is STL only on this review branch. It becomes
the repository default only after the complete-X2 human review is accepted.
