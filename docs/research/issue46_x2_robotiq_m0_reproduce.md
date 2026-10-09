# Issue #46 X2 + Robotiq M0 Reproduction

This reproduces the static preflight only. The runner stops before dynamics when the initial collision or source-limited pregrasp IK gate fails.

## Runtime

- WSL Ubuntu 22.04
- Python 3.10.12 with MuJoCo Python/native 3.3.6
- `MUJOCO_GL=egl`
- X2 checkout at `/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf`, clean at `575cc6b988f976c23550e0db85aa1e5475d3652d`
- Menagerie checkout at `/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie`, clean at `0059d4335f8156206f63a35662313385f7ad6d74`
- Canonical helper at `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py`, SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`

## Commands

Run candidate 1:

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009 && MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_x2_robotiq_m0.py --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf --menagerie-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-x2-simulation-only-m0-20261009/candidates/station_candidate_1 --station-base-pos -0.08 0 0.68
```

Run candidate 2:

```bash
cd /home/lzy18001500226/robotsim-issue46-x2-robotiq-m0-20261009 && MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_x2_robotiq_m0.py --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf --menagerie-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-x2-simulation-only-m0-20261009 --station-base-pos 0 0.08 0.68
```

The script records the exact source pins, model and mesh hashes, runner hash, base pose, mount transform, model settings, initial contacts, head-pose scan, IK residuals, and failure stage in each output `result.json`. Do not interpret a static model screenshot or successful compilation as a grasp result.

## 2026-10-09 Station-Recovery Follow-up

The later mounted dynamic smoke and bounded two-configuration reachability checks are documented in [`issue46_x2_robotiq_m0_station_recovery_20261009.md`](issue46_x2_robotiq_m0_station_recovery_20261009.md). They stop before bottle dynamics because the mounted OPEN/CLOSE cycle has Robotiq coupler hard-limit excursions and neither collision-aware pregrasp passes. Reproduction commands and durable evidence are in the station-recovery report and its linked packet; this historical baseline is unchanged.
