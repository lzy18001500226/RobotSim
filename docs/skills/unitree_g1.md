# Technical note: Unitree G1

Public baseline:
- `unitree_sdk2`
- `unitree_ros2`
- `unitree_mujoco`
- G1 uses the `unitree_hg` low-level message family.

The official `unitree_mujoco` path is mainly a low-level Sim-to-Real simulator. Higher-level walking autonomy must be provided by another controller / policy / stack.
