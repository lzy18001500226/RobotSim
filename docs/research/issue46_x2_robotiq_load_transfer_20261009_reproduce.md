# Reproduce Issue #46 Isolated Load-Transfer Run

## Runtime

Use the pinned local Python 3.10.12 environment containing MuJoCo 3.3.6 and set `MUJOCO_GL=egl`:

- Python: `/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python`
- Menagerie checkout: `/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie`
- Canonical helper: `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py`
- Script: `scripts/research/issue46_x2_robotiq_isolated_bottle.py`

The corrected final run used script SHA-256 `25b5eee232d5a74ef333e9d2fdeefee345b5b71fcffe367e0a439d33d52a887b`, Menagerie model SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`, and canonical helper SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.

## Command

Run from the RobotSim checkout. This reproduces the bounded final candidate into a new directory; do not replace `grasp_com_z15_clearance13_verified/`, which contains the corrected, preserved run.

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_x2_robotiq_isolated_bottle.py \
  --menagerie-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie \
  --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py \
  --pad-midpoint-offset-world-m -0.012 0 0.015 \
  --carriage-bias-compensation \
  --carriage-servo-stiffness-scale 25 \
  --airborne-clearance-target-m 0.0013 \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-load-transfer-20261009/reproduction
```

This is the **SIMULATION_ONLY diagnostic candidate**, not a source-faithful success. The carriage servo uses the measured startup bias as target feedforward; stiffness is 25x the original fixture-servo value and damping is scaled by its square root; the 25 N force limit remains. The pad midpoint is offset `(-12, 0, +15) mm` from the canonical bottle body center. Bottle and gripper physical properties and source joint limits are unchanged.

## Read the Result

The process status `NON_ACCEPTANCE_PHYSICS_COMPLETE` means the bounded simulation completed; it does not mean the grasp passed. Read `result.json` and require all of the following before calling any lift a pass:

- `source_joint_limit_status == "PASS"`;
- the requested bottle COM rise is reached at the end of the hold;
- `table_contact_persistence_fraction == 0` for the whole hold;
- non-gripping bottle contact persistence is zero;
- bilateral pad contact persists and is present at the hold endpoint.

The preserved run stops at the first failed 1 mm airborne hold. It does not attempt 5/30/50 mm milestones or release. The full per-step records are `raw/isolated_physics_trace.jsonl` and `raw/isolated_contact_trace.jsonl`; `isolated_grasp_lift_release.mp4` is the runner's legacy filename and shows only the phases actually reached.

The separate `coupler_source_limit_ablation.json` is a one-step in-memory causal diagnostic: the comparison model neutralizes only the spring stiffness and is not a production or acceptance model.
