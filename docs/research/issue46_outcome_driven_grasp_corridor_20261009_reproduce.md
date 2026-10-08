# Issue #46 Outcome-Driven Grasp Corridor Reproduction

Run these commands from the isolated WSL2 research checkout. The first block records the exact invocations already used; those evidence directories are populated and must not be reused. The second block is for a fresh reproduction and writes to new output directories. The first run records both hypotheses; the second is the body-candidate evidence recovery with its complete OPEN/approach trace.

## Original Execution Commands

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-collision-audit-20261008
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_outcome_grasp_corridor.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/outcome-grasp-corridor-20261009/run-07

MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_outcome_grasp_corridor.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/outcome-grasp-corridor-20261009/run-08 \
  --candidate body_side_near_horizontal_upper_pitch
```

## Fresh Reproduction Commands

These output paths are separate from the preserved evidence. The runner creates the leaf directories and refuses to overwrite nonempty output.

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-collision-audit-20261008
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_outcome_grasp_corridor.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/outcome-grasp-corridor-20261009/reproduction-20261009/run-07

MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_outcome_grasp_corridor.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/outcome-grasp-corridor-20261009/reproduction-20261009/run-08 \
  --candidate body_side_near_horizontal_upper_pitch
```

Inputs are pinned and checked by the runner:

- AgiBotTech X2 source commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Python `3.10.12`; MuJoCo Python/native `3.3.6`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Canonical scene helper SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.

The second run records outcome evaluator SHA-256 `64f602ee9dd79e4db854cb7c57bfeee47400f022bda7cda24d42ad9dd8cd8fe7`. The earlier run's result does not separately record the evaluator SHA. Each run calls `mj_forward` for collision/geometry queries but never calls `mj_step`; no physical contact forces or dynamic lift are measured. The root summary and raw result hashes are in `issue46_outcome_driven_grasp_corridor_20261009.md`.
