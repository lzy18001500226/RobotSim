# Reproduce Issue #46 Isolated Robotiq Lift and X2 Static Gate

The preserved run was performed from research branch `research/issue46-x2-robotiq-m0-20261009` at checkout HEAD `2a607c7e8bcd01fb93e18a12f1f3209fa7dcfa2c`, using Python 3.10.12, MuJoCo 3.3.6, and `MUJOCO_GL=egl`. Menagerie, X2 source, canonical-helper, model, and native-library identities are listed in the [experiment report](issue46_x2_robotiq_load_transfer_20261009.md).

The original executed argument arrays are retained in the relevant `result.json` files. The commands below reproduce the final isolated diagnostic and the three bounded static X2 candidates. Set `MUJOCO_GL=egl`, use the local Python/Menagerie/canonical-helper paths shown below, and choose new output directories so the preserved records are not overwritten.

## Isolated 50 mm Diagnostic

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_isolated_bottle.py \
  --menagerie-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie \
  --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py \
  --pad-midpoint-offset-world-m -0.007 0 0.015 \
  --carriage-bias-compensation \
  --carriage-servo-stiffness-scale 25 \
  --coupler-limit-activation-margin-rad 0.001 \
  --airborne-clearance-target-m 0.002 \
  --five-mm-carriage-target-m 0.008 \
  --large-lift-move-duration-s 1.0 \
  --thirty-mm-carriage-target-m 0.0305 \
  --fifty-mm-carriage-target-m 0.051 \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-load-transfer-20261009/reproduction
```

This is a **SIMULATION_ONLY diagnostic**. Its carriage servo, coupler activation margin, grasp offset, and overshoot commands are not source hardware parameters. It demonstrated 50.189 mm contact-driven bottle COM rise, bilateral contact, and no table support during the 1 s 50 mm hold. It does not demonstrate the full X2 arm or source-faithful acceptance. The preserved result is in `lift50_target_51mm/`.

## Integrated X2 Static Candidates

Run from the RobotSim checkout with the same Python and `MUJOCO_GL`:

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_reachable_grasp.py \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-load-transfer-20261009/x2_integration_calibrated_v1

MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_reachable_grasp.py \
  --candidate-name isolated_lift_calibrated_plus_y_approach \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-load-transfer-20261009/x2_integration_calibrated_plus_y_v1

MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_reachable_grasp.py \
  --candidate-name isolated_lift_calibrated_forward_station \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-load-transfer-20261009/x2_integration_forward_station_v1
```

These experiments are static only. The runner returns `BLOCKED_STATIC_REACHABILITY` and records `physics_run: false` for all three. Read each `result.json` and `*/raw/candidate_trace.json` for the exact solver iterations, qpos, signed distances, model identities, and command. Do not interpret these three candidates as a global reachability verdict.

## Preserved Evidence

External evidence directory:

```text
C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robotiq-load-transfer-20261009\
```

Important subdirectories are `lift50_target_51mm/`, `x2_integration_calibrated_v1/`, `x2_integration_calibrated_plus_y_v1/`, and `x2_integration_forward_station_v1/`. Do not edit the raw result JSON, JSONL traces, rendered images, videos, or generated model XML. `SHA256SUMS.txt` covers the preserved evidence packet.
