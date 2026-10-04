# Issue #43: G1/Humanoid Manipulation Reproducibility Audit

**Scope:** source-level audit of three upstream candidates for Local02. No RobotSim production code was changed and no policy was trained. Grades describe suitability as a reusable M0 manipulation baseline. A locomotion-only component is called out separately where useful.

## Executive result

There is no immediately reproducible released G1 pick-and-place policy in the three strongest sources inspected. Unitree's official Isaac Lab repository has the most concrete G1 pick/place scene, but its demo is driven by external DDS joint commands and cannot run in MuJoCo. The most reproducible G1 simulation/controller piece is the official Unitree RL Lab ONNX velocity policy paired with Unitree MuJoCo; it supplies stand/velocity locomotion, not a grasp task or actuated hand. HOVER is a useful H1 MuJoCo tracking reference, but has neither included policy weights nor object manipulation.

| Rank | Candidate | Grade | Local02 use |
|---|---|---|---|
| 1 | Unitree `unitree_sim_isaaclab` pick/place | **C — useful reference only** | BORROW TASK LOGIC |
| 2 | Unitree `unitree_rl_lab` + `unitree_mujoco` | **B for stand/velocity only; C for manipulation** | BORROW CONTROLLER |
| 3 | NVIDIA HOVER | **D — not worth pursuing for M0** | IGNORE |

## 1. Unitree official Isaac Lab G1 pick/place

**Repository and revision:** [unitree_sim_isaaclab, e30c25b](https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc) (2026-03-30). Official Unitree. Repository code is Apache-2.0. I found no policy checkpoint in the source tree and no distinct asset/policy license for the separately downloaded USD bundle; check its terms before redistribution.

### What actually launches

The README's two-finger-cylinder command is:

```bash
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129
```

It must run inside an Isaac Lab/Isaac Sim environment. `sim_main.py` starts `isaaclab.app.AppLauncher`; its default `--action_source` is `dds`. `create_action_provider.py` implements `dds`, `dds_wholebody`, and `replay`, but returns `None` for the advertised `file`, `trajectory`, and `policy` choices. The cited command therefore does **not** launch a pick/place policy: `DDSActionProvider` consumes externally published SDK2 joint/gripper commands. README points at `send_commands_8bit.py`/`send_commands_keyboard.py` and the separate `xr_teleoperate` project. There is a separate `dds_wholebody` path that loads `assets/model/policy.onnx` (the default external model path), but the pick/place command does not select it and it is not a grasp policy. `DDSRLActionProvider` uses that ONNX as a 12-output lower-body/ankle joint-target policy (scale 0.25); the 14 arm targets still come from DDS and the hand from its separate hand-command stream. Its observation is a 10-step history of body angular velocity, projected gravity, four run-command values, 26 lower-body/arm joint positions, 26 joint velocities, and 29 previous actions (expected 910-wide input). The 1 GiB+ external asset archive may carry the policy, but no ONNX/checkpoint is tracked in the Git tree and I did not download the archive to verify it. `--action_source=policy` is accepted by argparse but has no factory implementation. There is no pick/place policy wired into the cited demo.

### Model, task, controller, and contact

The task config selects `G1RobotPresets.g1_29dof_dex1_base_fix()`: a G1 29-DoF body preset with a bilateral Dex1 two-finger gripper (four hand joints in the scene). The README says only tasks labelled `Wholebody` provide mobile control; this `Joint` task is joint-space control, not a mobile whole-body policy. The USD is external: `assets/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd`.

The action is a direct all-joint position target (`joint_names=[".*"]`, scale 1, default offset enabled). The observation group is a non-concatenated set of body/joint state, gripper state, and camera images, rather than a fixed flat policy tensor. DDS maps 14 arm joint targets from body LowCmd positions 15–28 and reads gripper commands separately. In the preset, arm gains are shoulder Kp 25, elbow Kp 50, wrist Kp 40, with Kd 2; Dex1 hand gains are Kp 800, Kd 3, friction 200. Waist gains are set to 10,000/10,000 in this base-fix preset. The sim runs at dt=0.005 s with decimation 2 (100 Hz environment actions). These gains and the base-fixed asset need deliberate translation before reuse.

The cylinder is a real dynamic PhysX `RigidObjectCfg` (0.4 kg, collision enabled, friction 1.5); the source inspected contains no grasp/attach constraint. The task reward checks object position against a target region. This is physical contact in Isaac Sim, not a scripted object attachment, but a successful trajectory/controller is external. The scene also references an Isaac Nucleus warehouse and external table USDs.

### Dependencies and viability traps

- Docker pins Python 3.11, PyTorch 2.7.0 cu126, Isaac Sim 5.1.0, CUDA runtime 12.2, and CycloneDDS 0.10.x. It clones Isaac Lab from unpinned `main`; README also claims Isaac Sim 4.5/5.x support. The repo says it was tested on RTX 3080/3090/4090, not a 4060. The Torch cu126 wheel over a CUDA 12.2 base and unpinned Isaac Lab are reproducibility risks.
- `fetch_assets.sh` clones `unitree_sim_isaaclab_usds` from Hugging Face, requires `assets.zip` larger than 1 GiB, then extracts it. It deletes/replaces a sibling `assets` directory. The simulation is not a quick source-only checkout.
- Docker's documented launch needs GPU access, Vulkan ICDs, display/X11, host networking, and camera/streaming plumbing. `--device cpu` does not establish that Isaac Sim, PhysX, or camera rendering is CPU-only. WSL2 + RTX 4060 may work with a carefully configured WSLg/Vulkan/Docker stack, but is outside the stated GPU test matrix; DDS multicast/host-network behavior is another risk.
- The body/hand USD, PhysX gains, and camera stack do not load as the RobotSim-pinned MuJoCo MJCF. The external gripper has four driven joints while the official Unitree MuJoCo G1 model below exposes 29 actuators and static hand geometry.

**Recommendation:** BORROW TASK LOGIC only: cylinder mass/contact setup, target-region semantics, and task-specific observation ideas. Do not spend M0 time installing Isaac Sim to obtain a controller that is not included.

## 2. Unitree official G1 locomotion deployment paired with Unitree MuJoCo

This is one executable pipeline made from two official repos:

- [unitree_rl_lab, 4960b847](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3) (2025-11-19), Apache-2.0.
- [unitree_mujoco, 1eb6642](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d) (2026-09-07), BSD-3-Clause.

The committed G1 velocity ONNX is 1.6 MiB. There is no separate policy-weight license in the tree, so do not assume broader terms than the repository license without confirmation. Inference uses the bundled ONNX Runtime 1.22.0 C++ library on CPU; no CUDA/Torch runtime is required for this deployment path.

### What actually launches

The README starts `unitree_mujoco/simulate/build/unitree_mujoco`, then `unitree_rl_lab/deploy/robots/g1_29dof/build/g1_ctrl`. Configure both sides to DDS domain 0 and loopback `lo`. Use the `g1` robot and `scene_29dof.xml`. `g1_ctrl` uses Unitree SDK2 LowState input and LowCmd output. The gamepad/state sequence is: `[L2 + Up]` to enter FixStand, then `[R1 + X]` to enter Velocity. Keep the sticks neutral for zero command; the remote-control command is clamped by the policy config. The stale README keys `use_joystck` and `enable_elastic_hand` do not match MuJoCo's `use_joystick` and `enable_elastic_band`; keep the actual config names. The simulator XML parses at MuJoCo dt=0.002 s.

### Exact robot/controller interface

MuJoCo loads `unitree_robots/g1/scene_29dof.xml` including `g1_29dof.xml`: G1 29 actuated body joints (nq=36, nv=35, nu=29 with free root). A headless MuJoCo 3.3.6 parse succeeded. The fingers have no actuators in this model; the hand mesh is fixed. Thus it can carry/position wrists but cannot close a hand or produce a grasp.

The `velocity/v0` policy is trained from the Isaac Lab G1 29-DoF task, but its released ONNX is directly usable by the C++ deployer without Isaac training. ONNX Runtime CPU validation in a disposable venv reported input `obs [1,480]`, output `actions [1,29]`. Input is five samples of the ordered terms: base angular velocity (3), projected gravity (3), velocity command (3), joint position relative to defaults (29), relative joint velocity (29), and last action (29). Actions are 29 joint-position targets, scaled by 0.25 and offset by the configured default pose. `joint_ids_map` maps the policy ordering to the SDK2 LowCmd motor IDs. The config is 50 Hz (`step_dt=0.02`), or 10 MuJoCo 2 ms steps per policy update. Position-control gains in policy order are Kp `[100,100,100,150,40,40,100,100,100,150,40,40,200,200,200,40×14]` and Kd `[2,2,2,4,2,2,2,2,2,4,2,2,5,5,5,10×14]`.

Velocity command axes are read from the simulated joystick: forward `ly` clamped to [-0.5, 1.0] m/s, lateral `-lx` clamped to [-0.3, 0.3] m/s, yaw `-rx` clamped to [-0.2, 0.2] rad/s. Zero command is a trained standing case (the Isaac task assigns 2% standing environments), and leaving the policy in Velocity with centered axes is the intended stop behavior. FixStand is a separate initial mode; the Velocity FSM has no direct transition back to FixStand. A bad-orientation check transitions to Passive; there is no automatic get-up/recovery policy. Recovery is manual/reset-level.

### Reproduction procedure

This reproduces the **standing/velocity locomotion baseline only**, not manipulation:

1. In a disposable Ubuntu 22.04/WSL2 checkout, pin the three repos to the commits above and install `unitree_sdk2` at `63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36`. Install the README's native build dependencies: CMake, C++17 toolchain, Boost, yaml-cpp, spdlog, Eigen, fmt, and GLFW.
2. Download the official MuJoCo 3.3.6 Linux SDK to `~/.mujoco/mujoco-3.3.6`; symlink it as `unitree_mujoco/simulate/mujoco`. Install Unitree SDK2 to `/opt/unitree_robotics` as expected by `unitree_mujoco` CMake.
3. Build SDK2, then `cmake -S unitree_mujoco/simulate -B unitree_mujoco/simulate/build && cmake --build unitree_mujoco/simulate/build -j4`. Build `unitree_rl_lab/deploy/robots/g1_29dof` with `cmake -S ... -B .../build && cmake --build .../build -j`.
4. Set `unitree_mujoco/simulate/config.yaml` to `robot: g1`, `robot_scene: scene_29dof.xml`, `domain_id: 0`, `interface: lo`, `use_joystick: 1`, and `enable_elastic_band: 0`. Start `unitree_mujoco/simulate/build/unitree_mujoco -i 0 -n lo -r g1 -s scene_29dof.xml`.
5. From the RL Lab G1 deploy build directory, start `./g1_ctrl --network lo`. Enter FixStand, then Velocity as above. Verify neutral sticks hold a standing pose, forward/yaw tracks within configured limits, and neutral sticks stop translation/rotation. Do not infer a grasp capability from this result.

**WSL2 risk:** the CPU ONNX path and MuJoCo physics are light enough for an RTX 4060 laptop; GUI OpenGL/GLFW, DDS loopback, and SDK2 build/link setup are the likely friction points. MuJoCo model parse and ONNX CPU inference were checked here, but C++ build was not: this audit container lacks CMake and installed Unitree SDK2. No WSL2 machine or hardware validation was performed.

**Compatibility:** this is the best G1 29-body controller source and aligns with an SDK2 LowCmd/LowState boundary. It has no object assets, no wrist-target interface, and no active hand. Compare its joint ID mapping, default pose, servo gains, update cadence, and MuJoCo actuator/contact parameters against RobotSim before reuse; an exact middleware boundary does not make those contracts interchangeable.

**Recommendation:** BORROW CONTROLLER for a separate standing/velocity fixture, not as a manipulation policy or final architecture selection.

## 3. NVIDIA HOVER (H1 whole-body tracking)

**Repository and revision:** [NVlabs/HOVER, 8088f6c](https://github.com/NVlabs/HOVER/tree/8088f6cfb42a8f307dc614735197796a86ce8490) (2025-07-29). Community/research code under Apache-2.0. The nested Human2Humanoid/PHC material has separate CC-BY-NC-4.0 terms. No checkpoint license is applicable because trained policy weights are not included.

The simple viewer command is `${ISAACLAB_PATH}/isaaclab.sh -p neural_wbc/inference_env/scripts/mujoco_viewer_player.py`; it loads the H1 MuJoCo scene and pauses for manual stepping. It is not an autonomous manipulation demo. The actual policy evaluation command is the README's `eval.py` invocation with `--student_path` and `--student_checkpoint model_<iteration>.pt`. `DeploymentPlayer` requires this checkpoint. The README explicitly says `stable_punch.pkl` is not included and says no teacher checkpoint is provided. A useful student requires reference motion data plus teacher/student training in Isaac Lab; the eval physics wrapper is native MuJoCo, but the weights/data are not a ready download.

The source model is `neural_wbc/data/data/mujoco/models/scene.xml`, containing H1 (nq=26, nv=25, nu=19) plus floor; MuJoCo 3.3.6 headless parsing succeeded. The config uses 19 H1 joint controls, 0.005 s physics dt, decimation 4 (50 Hz), position targets scaled by 0.25 and converted to torque by Kp/Kd PD. Policy input is proprioception/history plus reference-motion tracking data; output is 19 joint-position actions. The reference's tracked “hands” are extended body points parented at the elbows, not articulated hand joints. There is no object scene, wrist-target API, or physical grasp. It is H1, not G1, with no finger DoF.

Dependencies: Python >=3.10, PyTorch, MuJoCo and viewer (unversioned in wrapper requirements), plus Isaac Lab 2.0.0 for the documented install/training environment; its Docker base is `nvcr.io/nvidia/isaac-lab:2.0.0`. Inference can use CPU tensors, but Isaac Lab teacher/student training expects GPU capacity. The HOVER README itself warns that the MuJoCo viewer can segfault due to OpenGL/driver/Conda mismatch and suggests EGL as a workaround. WSL2's viewer stack is therefore a predictable time sink; headless model loading is simpler. The H1 model, absent motion files/checkpoints, non-commercial third-party motion material, and no pick/place behavior make it a poor M0 pursuit.

**Recommendation:** IGNORE for Local02 M0. Keep only as a reference for reference-motion tracking and MuJoCo wrapper structure if that later becomes a requirement.

## Checks performed and evidence limits

- `unitree_mujoco` G1 29-DoF scene parsed headlessly with MuJoCo 3.3.6: `nq=36`, `nv=35`, `nu=29`, `dt=0.002`.
- HOVER H1 scene parsed headlessly with MuJoCo 3.3.6: `nq=26`, `nv=25`, `nu=19`, `dt=0.002`.
- Unitree G1 velocity ONNX loaded under ONNX Runtime 1.22.0 with `CPUExecutionProvider`; shapes were `[1,480] -> [1,29]`.
- The audit container has no Isaac Lab, PyTorch, CMake, or installed Unitree SDK2. The Isaac Sim task, native C++ simulator/controller build, WSL2 GUI/DDS behavior, and full task execution were not run. No policy was trained and no hardware was actuated.

## Evidence links

- [Unitree Isaac Lab README at audited commit](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/README.md), [`sim_main.py`](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/sim_main.py), [`create_action_provider.py`](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/action_provider/create_action_provider.py), [`DDS whole-body provider`](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/action_provider/action_provider_wh_dds.py), [Dex1 task config](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/tasks/g1_tasks/pick_place_cylinder_g1_29dof_dex1/pickplace_cylinder_g1_29dof_dex1_joint_env_cfg.py), [`fetch_assets.sh`](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/fetch_assets.sh), [`Dockerfile`](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc/Dockerfile).
- [Unitree RL Lab README at audited commit](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/README.md), [G1 velocity deploy config](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml), [C++ inference bridge](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/deploy/robots/g1_29dof/src/State_RLBase.cpp), [G1 MJCF](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1/g1_29dof.xml), [MuJoCo README](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/readme.md).
- [HOVER README at audited commit](https://github.com/NVlabs/HOVER/blob/8088f6cfb42a8f307dc614735197796a86ce8490/README.md), [MuJoCo wrapper README](https://github.com/NVlabs/HOVER/blob/8088f6cfb42a8f307dc614735197796a86ce8490/neural_wbc/mujoco_wrapper/README.md), [MuJoCo evaluation script](https://github.com/NVlabs/HOVER/blob/8088f6cfb42a8f307dc614735197796a86ce8490/neural_wbc/inference_env/scripts/eval.py), [H1 config](https://github.com/NVlabs/HOVER/blob/8088f6cfb42a8f307dc614735197796a86ce8490/neural_wbc/inference_env/inference_env/neural_wbc_env_cfg_h1.py).
