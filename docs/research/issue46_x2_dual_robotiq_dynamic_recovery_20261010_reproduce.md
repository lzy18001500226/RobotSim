# Reproduce the Dual Robotiq Recovery Checks

These commands use the exact branch checkout, pinned Python/MuJoCo runtime, controller helper snapshot, and vendor source checkouts recorded in the result JSON. Choose fresh, empty output paths; do not point them at the preserved evidence directories.

Set `EVIDENCE_DIR` to the root of the accompanying local evidence packet before running these commands. The packet contains the controller helper snapshot and canonical scene helper.

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_coupler_recovery.py \
  --output-dir "${TMPDIR:-/tmp}/robotsim-issue46-dual-robotiq-recovery-replay-source" \
  --controller-helper-snapshot "${EVIDENCE_DIR}/source/snapshots/issue46_x2_robotiq_m0_5954a772.py" \
  --menagerie-xml /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie/robotiq_2f85/2f85.xml \
  --coupler-margin-rad 0.001
```

This rebuilds the dual assembly from pinned source, records the source-faithful one-step failure and the margin-only diagnostic, then runs no-object cycles without the head/torso exclusion. It is expected to stop those cycles at the verified source head/torso overlap and head-pitch limit.

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_coupler_recovery.py \
  --output-dir "${TMPDIR:-/tmp}/robotsim-issue46-dual-robotiq-recovery-replay-diagnostic" \
  --controller-helper-snapshot "${EVIDENCE_DIR}/source/snapshots/issue46_x2_robotiq_m0_5954a772.py" \
  --menagerie-xml /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie/robotiq_2f85/2f85.xml \
  --coupler-margin-rad 0.001 \
  --exclude-known-head-torso-source-contact \
  --canonical-helper "${EVIDENCE_DIR}/source/canonical_manipulation_assets.py"
```

The second command reproduces the SIMULATION_ONLY dynamic left/right/both cycles and the static 101-sample canonical workcell clearance check. The right-wrist/follower collision and wrist/bottle penetration remain expected failures. It does not run bottle physics.

Runtime identity: Python `3.10.12`; MuJoCo `3.3.6`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`. The preserved canonical helper SHA-256 is `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`. The second command uses runner SHA-256 `2d9effdb3aa69a00dc1e4c1888764fb8c424ec78cbe7f1d23e4843638f88d752`. The earlier no-exclusion trial records runner SHA-256 `6ac4fa758e82991d247233e791ca704df9fd4bdbf9db5631ac4db04802c0cf0a`; that historical runner source snapshot is not retained, so the first command is a source-rebuild reproduction using the preserved current runner, not a byte-identical replay of that earlier invocation.
