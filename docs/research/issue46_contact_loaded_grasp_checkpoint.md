# Issue #46 Contact-Loaded Grasp Checkpoint

**Result: CHECKPOINT FAIL - contact-loaded mimic tolerance exceeded.** This bounded run stopped after the short lift attempt. It did not run transfer, lower, release, or settle.

## Runtime and Scope

- RobotSim branch/head used by the run: `codex/issue46-lingxi-scene-frames` / `2ef307f02cd0524833e8fbdf29a69072c1c4d6cc`.
- Official source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`, model `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`, Mulan PSL v2.
- URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`; referenced mesh hashes are recorded in `runtime_identity.json`.
- Python `3.10.12`, MuJoCo `3.3.6`, NumPy `1.26.4`, timestep `0.002 s`, implicit-fast integrator.
- Canonical bottle: `0.57 kg`, `70 mm` diameter, `244.5 mm` height. The only grasp-pose adjustment was the previously measured `+0.007 m` palm-Z correction. The accepted `kp=0.1`, `kv=0.003` mimic-driver profile and `0.003 rad` tolerance were used; no friction or gain tuning was performed.
- Rollout: 3,000 steps / 6 simulated seconds: approach, close, grasp approach, 1-second hold, and a 2-second lift command targeting `30 mm`.

## Findings

- **Physical integrity:** bottle remained a free MuJoCo body. The model had 12 equalities, all source hand mimic joints; no bottle weld/equality, mocap, follow-hand logic, or active-rollout bottle-qpos write occurred. The opposite hand made no bottle contact. Bottle qpos was unchanged during robot initialization.
- **Approach:** no right-hand/bottle contact and no right-hand/table contact before close. Bottle translation through pregrasp was `0.182 mm`, below the existing `2 mm` gate.
- **Contact topology:** first contact was `R_ring_pip` at `2.946 s`. During hold and lift, the exact contacting links were `R_middle_pip` and `R_ring_pip` (middle and ring families). There was no thumb contact, so the required opposing thumb grasp was not formed.
- **Hold:** the target arm command was stationary for `1.0 s`; bottle slip was `4.385 mm`, orientation change `0.023410 rad` (`1.341 deg`), right-hand contact persistence `100%`. The measured arm-joint change norm was `0.05117 rad`, and wrist displacement was `6.180 mm`. From hand-only pairs in the raw trace, normal force averaged `0.868 N` and peaked at `1.224 N` across the hold.
- **Mimic gate:** first error above `0.003 rad` was `L_thumb_pip_joint` at step 1 / `0.002 s` during approach: `0.00561686 rad`, before bottle contact. During the close phase, the first contact-window exceedance was `L_thumb_dip_joint` at step 501 / `1.002 s`: `0.08103341 rad`. At first bottle contact the maximum relation error was `0.08130171 rad`. Maximum across close, grasp approach, hold, and lift was `0.20840350 rad` during grasp approach. Per-phase maxima and all 12 relations are in `hand_mimic_trace.jsonl`.
- **Other hand-state checks:** `3,141` joint-limit violation samples, first observed at step 1; `138` mimic sign-inversion samples; zero NaN samples. No limit or mimic thresholds were relaxed.
- **Short lift:** maximum bottle rise during the lift phase was `0.08025 mm` versus the `30 mm` target. Middle/ring contacts persisted for all 1,000 lift steps, but the bottle remained supported by the table and was not carried.
- **Penetration and continuity:** maximum measured bottle-contact penetration was `1.206 mm`; maximum per-step bottle translation was `0.0573 mm`. No discontinuous bottle-state write was used.

The first failed condition is the mimic tolerance breach at step 1. The later failures are missing thumb opposition and failure to carry the bottle through the short lift. This physical run is new evidence that the previously passing free-space mimic profile does not preserve its gate in the integrated gravity-loaded rollout. No further grasp tuning or later manipulation phase was run.

## Evidence

External evidence directory: `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/checkpoints/20261005-contact-loaded-short-lift/`.

The directory contains `checkpoint_result.json`, `checkpoint_contact_trace.jsonl`, `hand_mimic_trace.jsonl`, `bottle_pose_trace.jsonl`, `runtime_identity.json`, `run.log`, `reproduction_command.txt`, the approach/first-contact/closed-grasp/hold/maximum-lift PNG frames, and `contact_loaded_short_lift.mp4`. The video was decoded successfully as H.264, 1280x720, 25 fps, 153 frames.

Exact reproduction command:

```bash
set -o pipefail
mkdir -p /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/checkpoints/20261005-contact-loaded-short-lift
cd /tmp/robotsim-issue46-lingxi-scene-frames
MUJOCO_GL=egl ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/checkpoints/20261005-contact-loaded-short-lift ISSUE46_FINGER_CLOSE_FRACTION=1.0 ISSUE46_GRASP_PALM_Z_OFFSET_M=0.007 ISSUE46_CHECKPOINT_LIFT_M=0.030 /tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python scripts/research/issue46_x2_contact_checkpoint.py 2>&1 | tee /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/checkpoints/20261005-contact-loaded-short-lift/run.log
```

Maintainer review is required before any further grasp tuning or continuation beyond the short-lift checkpoint.
