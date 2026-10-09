# Reproduce Issue #46 X2 + Robotiq Reachable Grasp Checkpoint

Run from WSL Ubuntu 22.04. These commands use the exact X2, Menagerie and canonical-helper pins recorded in the report.

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009
PY=/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python
OUT=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-reachable-contact-20261009
X2=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf
MEN=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie
CANON=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py

MUJOCO_GL=egl "$PY" scripts/research/issue46_x2_robotiq_coupler_init_check.py \
  --x2-root "$X2" --menagerie-root "$MEN" --canonical-helper "$CANON" \
  --output-dir "$OUT/coupler-startup-correction" --station-base-pos -0.08 0 0.68

MUJOCO_GL=egl "$PY" scripts/research/issue46_x2_robotiq_reachable_grasp.py \
  --x2-root "$X2" --menagerie-root "$MEN" --canonical-helper "$CANON" \
  --output-dir "$OUT/grasp-frames-pitch-correction"

for candidate in \
  outboard_station_plus_x_upper_body \
  outboard_station_minus_y_body_center \
  outboard_station_plus_x_lower_body; do
  MUJOCO_GL=egl "$PY" scripts/research/issue46_x2_robotiq_clearance_replay.py \
    "$OUT/grasp-frames-pitch-correction/$candidate"
done
```

The coupler command executes one 1,900-step mounted OPEN/CLOSE/REOPEN diagnostic with the single tested initial-qpos correction. It is expected to report `FAIL_MOUNTED_CYCLE` because the couplers exceed source hard limits. The grasp-frame command is zero-step IK/collision analysis only and is expected to report `BLOCKED_STATIC_REACHABILITY`. The clearance replay only evaluates `mj_geomDistance` at the saved joint poses; it does not rerun IK or step physics.

The source-versus-mounted head/torso comparison and pre-correction 1,900-step trace are preserved under:
`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-x2-head-contact-smoke-20261009/attempt_04/`.
