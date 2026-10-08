# Reproduce: Issue #46 OmniPicker Grasp Topology Study

## Preconditions

- WSL2 Ubuntu 22.04.
- Python 3.10.12 environment at `/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python` with MuJoCo 3.3.6.
- Pinned vendor checkout at `/tmp/robotsim-issue46-virtual-transmission-vendor-20261007`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Stage 1 corrected elbow-root source audit at `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-collision-audit-recovery/run-20261008-i`.

## Command

From the repository worktree:

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  /home/lzy18001500226/robotsim-issue46-omnipicker-collision-audit-20261008/scripts/research/issue46_omnipicker_grasp_topology_redesign.py \
  --stage1 /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-collision-audit-recovery/run-20261008-i \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-grasp-topology-redesign/run-20261008-3
```

The output directory must be absent or empty. The final recorded command redirected combined output to `/tmp/robotsim-issue46-omnipicker-grasp-topology-run-20261008-3.log`; the same log is copied into `run-20261008-3/run.log`.

The runner returns exit status 1 when all static candidates fail. For this recorded run that status is the expected experiment verdict, not a Python/runtime crash. It does not start physics after a static failure.

## Focused checks

```bash
python3 -m py_compile scripts/research/issue46_omnipicker_grasp_topology_redesign.py
git diff --check
```

No general test suite or physics rollout was run for this static-only research runner.
