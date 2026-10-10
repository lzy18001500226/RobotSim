# Reproduce: Dual Robotiq X2 Platform

This reproduces the bounded SIMULATION_ONLY dual-gripper model and its structural/initial-dynamics gates. With the currently pinned source model and controller, the expected terminal result is `BLOCKED` at the first physics step because the Robotiq coupler joints exceed their unchanged source upper limits. Bottle manipulation is not part of this run.

## Requirements

- WSL2 Ubuntu environment used for RobotSim Issue #46.
- Python environment with MuJoCo 3.3.6, NumPy, SciPy, Pillow, and imageio/ffmpeg.
- Cached source repositories:
  - X2: `/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf` at `575cc6b988f976c23550e0db85aa1e5475d3652d`.
  - Menagerie: `/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie` at `0059d4335f8156206f63a35662313385f7ad6d74`.
- Canonical asset helper: `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py`, SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Imported RobotSim helper: `scripts/research/issue46_x2_robotiq_m0.py`, SHA-256 `5954a772be1bede09b7c8f25425b61ddce809516cf316d77ecd5fd11cbbeb063` for the recorded run. The file was already modified by separate transfer work and is intentionally not included in this task commit; the committed `HEAD` version hashes to `e6ddac0c1ffe78031d1b14be094bc16c5a1501d5179b1167b830bf3a30c39c06`. The recorded run did not use a clean RobotSim worktree. For bit-for-bit reproduction, use the recorded working-tree copy; `result.json` stores the exact repository status and both helper hashes.

## Command

From the RobotSim repository root:

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_dual_robotiq_platform.py \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/dual-robotiq-platform-20261010
```

The script writes the generated XML, result JSON, raw traces, static images, and videos only to the output directory. Its exit code is `2` when a gate blocks completion; that is the expected current outcome. Do not interpret the kinematic closed-pose render as dynamic jaw motion. The current physics clip ends at the first failed step.

## Expected result

`result.json` should report:

- `status`: `BLOCKED`.
- Structural inventory: two distinct gripper namespaces; every instantiated imported body, joint, geom, mesh, actuator, tendon, equality and site name is side-prefixed and unique; X2 source body/joint tree retained except the fixed-base free joint; all retained X2 joint types, axes, limited flags and ranges match source; both wrist roll joints retain their motors; no X2 five-finger hand subtree; initial wrist mount clearances pass.
- Dynamic gate: `FAIL` at step 1 / 0.001 s with four Menagerie coupler joints slightly above the unchanged 0 rad upper limit.
- Independent left and right clean-reset trials: each fails before jaw closure on the same first-step limit condition.
- `qpos_writes_after_rollout_start`: `0`.
- Bottle contact and pick-place: not run.
- The RobotSim source head at run time was `1588e3dca37772fc3ad30335259ba6d44485ee89` on `research/issue46-x2-robotiq-m0-20261009`.

Do not tune, clamp, or alter the source coupler limits under this reproduction instruction. Any proposed fix requires a separate authorized task and must preserve this run as the baseline.
