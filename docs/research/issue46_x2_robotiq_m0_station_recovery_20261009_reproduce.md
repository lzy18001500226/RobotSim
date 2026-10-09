# Issue #46 X2 + Robotiq Station Recovery Reproduction

WSL Ubuntu 22.04; Python 3.10.12; MuJoCo Python/native 3.3.6; `MUJOCO_GL=egl`. The commands below reproduce the mounted smoke and bounded static reachability checks. No bottle contact rollout is part of this packet.

Pinned checkouts:

- X2 root: `/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Menagerie root: `/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie`, commit `0059d4335f8156206f63a35662313385f7ad6d74`.
- Canonical helper: `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py`.
- Environment: `/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python`.

## Mounted OPEN/CLOSE

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_x2_robotiq_mounted_smoke.py \
  --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
  --menagerie-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie \
  --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-x2-head-contact-smoke-20261009/attempt_04 \
  --station-base-pos -0.08 0 0.68
```

This exact command produced the retained attempt 04. The functional jaw cycle completed, but the result status is `FAIL` because both Robotiq coupler joints exceeded their upper hard limit by up to `0.000150054301 rad`.

## Source-Limited Reachability

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_x2_robotiq_reachability.py \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-x2-reachability-20261009/run_03
```

The runner evaluates only the two fixed configurations listed in its source and writes every optimizer evaluation to each candidate's `raw/candidate_trace.json`. Run 01 is the original wrist-orientation-seeded diagnostic, run 02 starts from the clear neutral arm, and run 03 adds the bounded position-only FK seed followed by collision-aware full-pose IK. All three runs have separate output directories; do not overwrite them.
