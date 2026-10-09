# Reproduction: Issue #46 Mounted X2 + Robotiq 50 mm Pickup

The command below is the command used for the final mounted 80 mm TCP path trial. It writes the raw output directory shown, so choose a new output directory before rerunning if the preserved evidence must remain untouched.

```bash
set -o pipefail
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009/scripts/research/issue46_x2_robotiq_m0.py --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/scene-first-workcell-20261009/layout_02_north_edge_80mm_evidence_derived_lift --station-base-pos 0.30 0.40 0.68 --station-base-yaw-deg -90 --table-center-xy 0.30 -0.10 --bottle-root-pos 0.30 0.045 0.9175 --target-bottle-root-pos 0.30 0.020 0.9175 --grasp-height-offset-m 0.050 --pregrasp-retreat-m 0.020 --lift-command-m 0.080 --coupler-limit-activation-margin-rad 0.001 2>&1 | tee /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/scene-first-workcell-20261009/layout_02_north_edge_80mm_evidence_derived_lift_run.log
```

## Tested Identity

- Runner SHA256: `68cc5ae6f4b21ca61199f9841c0e2b28b48fae0caaeda9158e2b07fff87f526b`.
- Python `3.10.12`; MuJoCo Python and native `3.3.6`; `MUJOCO_GL=egl`.
- X2 source commit: `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- MuJoCo Menagerie commit: `0059d4335f8156206f63a35662313385f7ad6d74`.
- Native MuJoCo library SHA256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Result: mounted `PARTIAL_PASS`; 56.295 mm peak and 54.224 mm minimum bottle COM lift during the one-second airborne hold, followed by return and release at the start position. Transfer/place was not run because source-limited transfer/place IK failed.

This reproduction is `SIMULATION_ONLY_DIAGNOSTIC`; it uses a 1 mm simulation-derived coupler activation margin and excludes only the verified internal `head_pitch_link`/`torso_link` contact pair. No bottle qpos writes, weld, mocap/follow, hidden support, or opposite-hand assistance were used. It is not source-faithful acceptance or hardware validation.

The separately captured isolated fixed-mount pickup was independently verified in `isolated_lift_independent_verification.json`; do not rerun it to interpret this mounted trial.
