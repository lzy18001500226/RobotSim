# Reproduce the Issue #46 Mount Audit

This command reproduces the corrected morphology, static-clearance, and no-contact gripper open/close diagnostic. It does not run bottle manipulation.

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_mount_replacement.py \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/mount-replacement-20261010/attempt-03
```

The output records the X2 and Menagerie source pins, native library/model hashes, compiled body/joint/geom/actuator inventories, initial collision audit, open bottle clearances, gripper joint/effort/contact traces, static views, and MP4. A nonzero exit is expected while the wrist/bottle clearance, wrist/gripper contacts, or source coupler-limit violations remain.
