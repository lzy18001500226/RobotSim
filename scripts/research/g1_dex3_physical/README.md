# Isolated G1/Dex3 physical baseline (research, NOT a successful pickup)

Current measured result: physical thumb/index/middle CONTACT_HOLD passes; LOAD_READY fails in two bounded attempts. No airborne lift or placement is validated. DFQ / PR45 are independent and unchanged. Do not interpret conditional later-phase code as tested pick-and-place.

Requires the persistent official Unitree checkout at 5994d4faef0a9cadd3287f8de0199a67eeb2a259 and preexisting G1/Dex3 20261015 static-review XML. `baseline.py` declares local paths; these are intentional research dependencies, not a portable fresh-clone launcher. MuJoCo 3.3.6, numpy, scipy, OpenCV, EGL are needed. No ROS/Isaac installation.

Official model: unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml. BSD-3-Clause; upstream assets remain external. No meshes redistributed here. Seven independent joints per hand, no mimic. Name mapping avoids SDK/MJCF index-order differences. Original mesh scale, collisions and mounts remain unchanged. Fixed-base, body gravity compensation and PD actuator formulation are simulation assumptions. Hand kp=1.5, kd=0.1 comes from official SDK2 example, not calibrated hardware dynamics.

Evidence (including report, two MP4s, original executed snapshots and source hashes):
`/home/lzy18001500226/projects/RobotSim/reviews/issue-43/dex3-physical-20261016/`

From this worktree in WSL:
```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/venv/bin/python scripts/research/g1_dex3_physical/test_contract.py
# Only when explicitly reproducing, choose a NEW run ID; failure exits nonzero.
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/venv/bin/python scripts/research/g1_dex3_physical/episode.py --run-id dex3-reproduce-attempt02-new --attempt 2
```

`plan.py` computes the bounded source-valid contact initialization using mj_forward only; `episode.py` executes actual arm/finger dynamics. Object qpos writes only occur in initialization/IK/planning on disposable data, never the active rollout. `summarize.py`, `audit_saved_states.py`, `freeze_identity.py`, `postprocess.py` analyze saved evidence. Offline reconstruction writes are not physical trials.

The 0.570 kg four-geom bottle, friction and source effort caps are unchanged. Both trials maintain bilateral contact, but measured mean hand support is only 0.00244 / 0.25967 N versus bottle weight 5.5917 N. No friction increase is justified by this result alone. Stop-loss reached; no additional parameter sweep.
