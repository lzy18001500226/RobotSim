# Reproduction

Run from WSL2 Ubuntu 22.04 in the isolated research checkout. The output directories must be empty.

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-collision-audit-20261008
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_source_faithful_grasp_corridor.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/grasp-corridor-20261008/run-04

MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_source_triangle_clearance.py \
  --result /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/grasp-corridor-20261008/run-04/result.json \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/grasp-corridor-20261008/run-04/source-surface-validation
```

The first command evaluates only the two source wrist-pitch endpoint hypotheses and records the compiled MuJoCo collision distances. The second checks the original collision STL surfaces at those saved OPEN/contact poses. Neither command calls `mj_step`; neither searches additional poses.

Pinned inputs are verified by the runner: AgiBotTech X2 commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, canonical manipulation helper hash `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`, and centered-jaw reference frame hash `0fdc574d47a0b08c440d06b96bffa0a00f8fe119fedad4d7a5eae108ff708f49`. Runtime and collision mesh hashes are recorded in `result.json`.
