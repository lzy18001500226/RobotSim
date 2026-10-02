# Workflow: G1 Baseline

## Objective
Verify the official Unitree MuJoCo path before adding perception or navigation.

## Steps
1. Fetch `unitree_sdk2`.
2. Fetch `unitree_mujoco`.
3. Build and install SDK2.
4. Link MuJoCo 3.3.6 into `unitree_mujoco/simulate/mujoco`.
5. Build the C++ simulator.
6. Change simulator robot to `g1`.
7. Use loopback (`lo`) and simulation DDS domain.
8. Start G1 simulation.
9. Verify state messages.
10. Only after this is stable, add grasp control.

## Acceptance criteria
- G1 MJCF loads.
- Simulator runs without crashing.
- G1 uses `unitree_hg` low-level messages.
- LowState can be observed.
- A small controlled joint command produces the expected motion.
