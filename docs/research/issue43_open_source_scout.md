# Issue #43: Open-source G1 MuJoCo pick-and-place reuse scout

**Reviewed:** 2026-10-04
**Purpose:** Find the shortest honest path to a fixed/planted G1 picking a bottle-like cylinder and placing it at a target. Free walking, learned perception, and a VLA are not prerequisites.

I searched GitHub beyond Unitree-owned repositories and inspected source, model files, license/notice files, dependency manifests, task runners, and physics-step paths at the commit IDs below. Tag queries returned no tags for the five candidates; the IDs below are inspected heads. This is a targeted scout, not a claim that every GitHub project was examined. I did not execute a candidate: this environment has no Python mujoco, cv2, or imageio modules, so README success numbers below remain author-reported.

## Ranked top 5

Only the first three contain a complete scripted physical pick-and-place task. Candidates 4–5 are useful supporting references, not substitutes for that task.

| Rank | Repository and inspected head | Status/license | Fit |
|---|---|---|---|
| 1 | [lwm97/pickandplaceunitreeg1](https://github.com/lwm97/pickandplaceunitreeg1/tree/b07678b131885c55175e23d2342548263eaeb9b0) | Community; MIT project code, separate BSD-3-Clause G1 asset notice | Closest licensed native-MuJoCo cylinder-to-pad demo |
| 2 | [RayyyyyQi/unitree-g1-mujoco-pick-place](https://github.com/RayyyyyQi/unitree-g1-mujoco-pick-place/tree/1b036cee9540a52e62ec2ef43718b89de8467e21) | Community; no project license found | Strong deterministic physical cube-to-tray task; reuse is not licensed and MuJoCo pin is 3.12 |
| 3 | [Indiguana/g1-manipulation](https://github.com/Indiguana/g1-manipulation/tree/25469c51c8b313fd4d8e42c9f501e3b480af271d) | Community; Unitree assets BSD-3-Clause, controller-code grant unclear | Physical pick/place and useful symbolic task/data path; MuJoCo 3.12 |
| 4 | [luckyrobots/g1-manipulation-challenge](https://github.com/luckyrobots/g1-manipulation-challenge/tree/0e9d1f9b61772905c9f7997b527f221dc7800fc9) | Community; no license found | Exact cylinder/table scene and ONNX reach/walk policies, but no automated placement pipeline |
| 5 | [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b) | Official; BSD-3-Clause | Pretrained G1 locomotion and MuJoCo deployment only; no hand or manipulation task |

## Candidate details

### 1. lwm97/pickandplaceunitreeg1 — primary

- **Activity/ref:** Community repository; inspected head b07678b131885c55175e23d2342548263eaeb9b0 (2026-09-04), “changes for multiscene”; no tag found.
- **License:** Root project license is MIT. NOTICE and third_party/unitree_g1/LICENSE separately identify Unitree/MuJoCo Menagerie G1 model/mesh material as BSD-3-Clause. Preserve both notices and follow RobotSim's asset-specific redistribution caution.
- **Robot/task:** g1_robot.xml is a fixed seated g1_29dof_with_hand_rev_1_0: 29 G1 body joints plus 14 Dex3-style finger joints (43 actuated DoF; no floating-base DoF). g1_scene.xml has a free, roughly 30 mm diameter × 90 mm, 34 g cylinder and a physical target pad. The active run_simulation.py path performs approach, descend, close, lift, transfer, lower, release. The object is stepped under MuJoCo contacts; the active LightWAMPolicyController uses bilateral fingertip force regulation and has no object weld/teleport during the rollout. The optional unused fastwam_policy.py contains an equality-constraint grasp path; do not treat that optional path as evidence of physical grasping.
- **Control/weights:** Native MuJoCo Python and mj_step, staged damped-least-squares IK plus position control and tactile grip. No learned weights are required; PyTorch is optional for an unused legacy Light-WAM loader. No CUDA/training requirement.
- **Runtime/headless:** Python 3.10+, mujoco>=3.1.0, NumPy, OpenCV, ImageIO, imageio-ffmpeg, Pillow. The --no-render flag suppresses output video, but the active controller still creates offscreen MuJoCo renderers for camera-based HSV detection/back-projection. Headless use therefore needs a working GL backend such as EGL or OSMesa. A laptop RTX 4060 is ample in principle for this small scene; CPU inference is enough and GPU is optional (the five-scene video benchmark benefits from it). WSL2 offscreen GL setup is the main unverified friction; this scout did not benchmark it.
- **Reproduction (README command, pinned head added):**

      git clone https://github.com/lwm97/pickandplaceunitreeg1.git
      cd pickandplaceunitreeg1
      git checkout b07678b131885c55175e23d2342548263eaeb9b0
      python3 -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt
      python run_simulation.py --no-render

  For a headless WSL2 attempt, configure a supported MUJOCO_GL backend before the last command. There is no seed flag in the documented single-run command. README reports 5/5 placements and 28–34 mm errors, but output JSON/video are gitignored; that result was not reproduced here.
- **RobotSim reuse/adaptation:** Best baseline for phases, DLS arm tracking, contact/tactile grasp, and release checks. RobotSim's pinned Unitree model is the 29-DoF body without Dex3 fingers; do not drop in this candidate's full robot model/assets. Adapt the control pattern to the pinned model and add a small RobotSim-owned gripper/contact model, initially fixed-base and ground-truth-pose driven. The candidate is visually richer than the minimum and its camera/CV route can wait.

### 2. RayyyyyQi/unitree-g1-mujoco-pick-place — secondary, pending reuse rights

- **Activity/ref:** Community repository; head 1b036cee9540a52e62ec2ef43718b89de8467e21 (2026-09-03), “Finalize submission README”; no tag found.
- **License:** No root license for the project controller/scripts was found. Unitree source/assets have their upstream license separately. Do not copy or adapt the author code without a license grant/permission.
- **Robot/task:** Fixed-base 29-DoF G1 body plus custom right Dex1-1 two-finger gripper (two additional finger joints). Native MuJoCo Python; models/task1/baseline.xml has a free-joint cube, table, and physical tray. The scripted task moves the cube only through MuJoCo contact and physics; qpos writes in the main flow are reset/initialization or replay paths, not task-time object attachment. It executes reach, grasp, lift, carry, tray contact, release and success checks.
- **Control/weights:** Hand-scripted joint targets, minimum-jerk ramps, IK, PD, contact checks. No learned controller/weights and no GPU.
- **Runtime/headless:** Python 3.10, exact lock pins mujoco==3.12.0, NumPy 2.2.6, SciPy 1.15.3, PyYAML, h5py, glfw, ImageIO and test/video packages. RobotSim pins MuJoCo 3.3.6, so use an isolated environment for upstream reproduction or separately port/test against 3.3.6; do not silently upgrade RobotSim's pin. CLI task is headless and CPU-realistic; viewer/video is optional. RTX 4060 is unnecessary.
- **Closest README reproduction:**

      git clone --recurse-submodules https://github.com/RayyyyyQi/unitree-g1-mujoco-pick-place.git
      cd unitree-g1-mujoco-pick-place
      python3.10 -m venv .venv
      source .venv/bin/activate
      python -m pip install -r requirements-lock.txt
      python scripts/smoke_test.py
      python -m pytest -q tests/test_level1_reset.py
      python scripts/full_pick_to_tray_center.py \
        --model models/task1/baseline.xml \
        --transport-wrist-roll-deg 90 \
        --place-descent \
        --coordinated-place
- **RobotSim reuse/adaptation:** Strong behavioral/test reference for deterministic reset, workspace variants, physical contact and place success criteria. First resolve code licensing, then assess the 3.12→3.3.6 API/model delta. Its generated model and Dex1 geometry are not RobotSim's pinned model.

### 3. Indiguana/g1-manipulation — third, pending code-license clarification

- **Activity/ref:** Community repository; head 25469c51c8b313fd4d8e42c9f501e3b480af271d (2026-09-05), “Language-conditioned pick-and-place…”; no tag found.
- **License:** Root BSD-3-Clause file/NOTICE attribute the Unitree model/assets. The NOTICE says the original g1_manip/ code is the author's own work, but the repository does not clearly grant a license for that controller code. Treat it as unavailable for copying until clarified.
- **Robot/task:** Fixed-base G1 29-DoF XML plus a runtime-attached two-slide parallel gripper (31 actuated DoF). Native MuJoCo Python/MjSpec and mj_step; dynamic free-joint blocks are picked and placed into a target area or bins with contact. The task calls body_xyz directly for object location; cameras are for demos/data, not a realistic detector. It is symbolic language conditioning (--color, --side), not a VLM/VLA. No trained policy/checkpoint is provided. It includes a 95-episode LeRobot-format demo path; that is a data pipeline, not evidence of a trained policy.
- **Control/results:** Closed-loop task-space DLS IK, joint PD, finger contact and scripted phases. README reports 43/50 for the single-block task and 52/52 for the conditional task, not independently reproduced. test_grasp/check_env are useful probes.
- **Runtime/headless:** Python 3.11 instructions; mujoco>=3.12, NumPy, Pillow, ImageIO. requirements.txt includes lerobot>=0.4 even though a comment says it is only needed for data conversion/verification; the README pip install command therefore pulls that dependency chain. The pick task itself only needs MuJoCo, NumPy, Pillow and ImageIO; install those core packages manually to avoid the optional LeRobot/PyTorch stack. CPU, no training GPU, and headless CLI are practical; renderer is created only when recording/viewing. Version mismatch with RobotSim 3.3.6 remains.
- **Closest README reproduction:**

      conda create -n g1 python=3.11 -y
      conda activate g1
      pip install -r requirements.txt
      python -m g1_manip.check_env
      python -m g1_manip.test_grasp
      python -m g1_manip.pick_conditional --seed 3 --color red --side left
- **RobotSim reuse/adaptation:** Useful architecture for a tiny gripper, closed-loop IK and keeping scene construction separate from the upstream G1 XML. The direct-state pose shortcut suits the requested first milestone but must later be replaced by RGB-D localization. Do not copy code before clarifying its license.

### 4. luckyrobots/g1-manipulation-challenge — scene and reach reference only

- **Activity/ref:** Community; head 0e9d1f9b61772905c9f7997b527f221dc7800fc9 (2026-04-07); no tag found.
- **License:** No repository license found, so no code/asset reuse without permission.
- **Robot/task:** Native MuJoCo scene with floating G1 29-DoF body plus 14 Dex3 finger DoF; free red cylinder, brown source table, blue target table. README presents pick/place as the challenge goal. Actual tracked run.py is a keyboard-driven viewer/control loop with pretrained walker/right-reacher ONNX policies and hand controls; there is no automatic approach→grasp→carry→release task or verified placement test. The cylinder is a physics body, but the repository does not supply a complete pick-and-place program.
- **Control/weights:** ONNX locomotion/reaching weights are included; inference does not require training or a GPU. The whole-body policy is not a manipulation policy.
- **Runtime/headless/adaptation:** README install is pip install mujoco onnxruntime numpy opencv-python with no version pins; run.py imports those packages and creates MuJoCo camera renderers. python run.py opens a viewer, and no headless task acceptance command is provided. The ONNX policies can run on CPU, but this scene/control loop was not checked on WSL2. Useful as a scene and reach-controller reference only, but unlicensed and floating/free-walk oriented, so not a first milestone.

### 5. unitreerobotics/unitree_rl_gym — future locomotion only

- **Activity/ref:** Official Unitree; head 276801e46c5d433564f24658bac64f254b7d2d4b (2025-07-25); BSD-3-Clause. No tag found.
- **Robot/task:** G1 29-DoF body, no hand/gripper. Includes pretrained deploy/pre_train/g1/motion.pt and native MuJoCo Sim2Sim deployment. The G1 config is a locomotion policy (12 actions/47 observations), not an arm manipulation or bottle-pick policy. No physical object task.
- **Runtime/headless/adaptation:** deploy/deploy_mujoco.py g1.yaml is the README path. setup.py requires Isaac Gym, rsl-rl, matplotlib, NumPy 1.20, tensorboard, MuJoCo 3.2.3, and PyYAML; the deployment also needs PyTorch. That MuJoCo version differs from RobotSim's 3.3.6. PyTorch policy inference is CPU-capable; training depends on the Isaac Gym/RL GPU stack. The documented deployment opens the MuJoCo viewer; headless task validation is not supplied. This is the strongest licensed future walking component, but adds no value while the G1 is fixed/planted.

## Other search results filtered out

- OyunbatBatzorig/g1-rgp-pickplace has an Isaac Lab multi-policy training path, but the needed reusable checkpoints are not released in the inspected tree and its README describes MuJoCo transfer losing the grasp within a few steps. This is longer and less reliable than scripted IK.
- ShrishChou/G1-D-Bimanual-Manipulation-and-Locomotion claims loco-manipulation, but its carried-object path writes the object's qpos along with the hand trajectory (teleport/kinematic carry); its G1-D wheeled model is not included.
- Zuckuu/g1-sim includes a MuJoCo roundtable scene, but its inspected path is described as a kinematic puppet and attaches the can by state update rather than a physical grasp.
- ghw1048040694/G1-VLA-Grasping did not include a usable G1 MJCF/checkpoint in the inspected tree. WIELD is a scaffold without a runnable released whole-body policy. derekc22/g1-diffusion depends on large external datasets/models and an older CUDA/PyTorch training stack rather than a ready MuJoCo bottle task.
- Official unitreerobotics/unitree_mujoco at RobotSim's pinned commit 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d is the correct licensed 29-DoF model/physics source, but contains no pick-and-place controller. RobotSim's third_party/LOCK.md warns not to assume every model/mesh can be redistributed; keep using the pinned checkout and verify specific asset rights.

## Recommendation

- **PRIMARY BASELINE:** lwm97/pickandplaceunitreeg1, because it is the shortest complete physical cylinder task with reusable MIT controller code, a documented BSD-3 asset notice, no training or GPU requirement, and a fixed seated base. First reproduce in isolation. For RobotSim, retain its pinned 29-DoF model; start with known cylinder pose and scripted approach/grasp/lift/place, porting only the DLS/contact-control pattern and adding a simple licensed gripper. Keep the object free and validate lift, carry, release, and final resting pose through MuJoCo dynamics.
- **SECONDARY FALLBACK:** RayyyyyQi/unitree-g1-mujoco-pick-place, for stronger reset/test determinism, only after receiving an explicit code license/reuse grant and validating its controller against MuJoCo 3.3.6.
- **THIRD FALLBACK:** Indiguana/g1-manipulation, for its symbolic task and data-collection structure, only after clarifying the original code license and handling the MuJoCo 3.12 requirement.
- Do not make a VLA, free walking, learned grasp policy, or RGB-D perception prerequisite. Use ground-truth pose only for initial integration; replace it with sensor perception before claiming realistic perception.

## Source anchors

- [RobotSim pinned G1 and MuJoCo provenance](https://github.com/lzy18001500226/RobotSim/blob/main/third_party/LOCK.md)
- [LWM README](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/README.md), [simulation_runner.py](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/simulation_runner.py), [lightwam_policy.py](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/lightwam_policy.py), [fastwam_policy.py](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/fastwam_policy.py), [license](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/LICENSE), [asset NOTICE](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/NOTICE)
- [Rayyyyy README](https://github.com/RayyyyyQi/unitree-g1-mujoco-pick-place/blob/1b036cee9540a52e62ec2ef43718b89de8467e21/README.md), [full_pick_to_tray_center.py](https://github.com/RayyyyyQi/unitree-g1-mujoco-pick-place/blob/1b036cee9540a52e62ec2ef43718b89de8467e21/scripts/full_pick_to_tray_center.py), [level1_env.py](https://github.com/RayyyyyQi/unitree-g1-mujoco-pick-place/blob/1b036cee9540a52e62ec2ef43718b89de8467e21/scripts/level1_env.py)
- [Indiguana README](https://github.com/Indiguana/g1-manipulation/blob/25469c51c8b313fd4d8e42c9f501e3b480af271d/README.md), [pick_place.py](https://github.com/Indiguana/g1-manipulation/blob/25469c51c8b313fd4d8e42c9f501e3b480af271d/g1_manip/pick_place.py), [scene.py](https://github.com/Indiguana/g1-manipulation/blob/25469c51c8b313fd4d8e42c9f501e3b480af271d/g1_manip/scene.py), [license](https://github.com/Indiguana/g1-manipulation/blob/25469c51c8b313fd4d8e42c9f501e3b480af271d/LICENSE), [NOTICE](https://github.com/Indiguana/g1-manipulation/blob/25469c51c8b313fd4d8e42c9f501e3b480af271d/NOTICE)
- [Lucky README](https://github.com/luckyrobots/g1-manipulation-challenge/blob/0e9d1f9b61772905c9f7997b527f221dc7800fc9/README.md), [run.py](https://github.com/luckyrobots/g1-manipulation-challenge/blob/0e9d1f9b61772905c9f7997b527f221dc7800fc9/run.py), [scene.xml](https://github.com/luckyrobots/g1-manipulation-challenge/blob/0e9d1f9b61772905c9f7997b527f221dc7800fc9/scene.xml)
- [Unitree RL-Gym README](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/README.md), [g1.yaml](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_mujoco/configs/g1.yaml)

## Confidence and handoff note

The GitHub REST API request to comment on RobotSim Issue #43 returned HTTP 403 at api.github.com/graphql. This report is therefore the durable fallback requested by the user. No RobotSim production code or vendor assets were modified, and no PR was created.
