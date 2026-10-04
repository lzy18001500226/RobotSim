# Issue #43 M0 Status

CURRENT STAGE: Phase 3 - build the RobotSim bottle/cylinder scene using the reproduced upstream G1/WBC runtime.

CHOSEN BASELINE: NVLabs HumanoidMimicGen.

UPSTREAM REPO/COMMIT: https://github.com/NVlabs/humanoidmimicgen @ d82844dcec242c82d6b82628ccc45933c3ad5cbd (main, inspected 2026-10-04).

WHY SELECTED: It is the strongest license-clear MuJoCo candidate found that already has a G1 table-to-table manipulation task, a three-finger hand model, recorded low-level action replay, physics-state regression, and a task-success predicate. Recorded playback runs on CPU; the WBC path is optional for baseline replay. Its current task object is a drill, so it is a controller/scene baseline rather than the final M0 object.

## Phase 1 Candidate Comparison

| Repository / commit | License | Robot / hand / simulator | Controller, assets, task | Reproduction and adaptation |
|---|---|---|---|---|
| [NVlabs/humanoidmimicgen](https://github.com/NVlabs/humanoidmimicgen) `d82844dcec242c82d6b82628ccc45933c3ad5cbd` | Apache-2.0 source with separately documented RoboCasa/RoboSuite MIT and Unitree BSD assets; downloaded WBC weights have NVIDIA Open Model License | Unitree G1, three-finger hand, MuJoCo 3.2.6 | Nine G1 tasks including Drill PnP; public recorded-action replay and task-success checks; WBC and task data are available separately | Python 3.10; core pins MuJoCo 3.2.6, NumPy 1.26.4, RoboSuite 1.5.1 and robosuite-models 1.0.0. Recorded replay is CPU-capable; public replay data is an external download. Cylinder adaptation is moderate/high. |
| [ozkannceylan/humanoid_vla](https://github.com/ozkannceylan/humanoid_vla) `3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12` | MIT; its G1 meshes come from a separate Unitree checkout | G1 29 DoF with palm pads, MuJoCo 3.x | Scripted Jacobian IK demos cover cube place; ACT checkpoint is not in the checkout and the single-arm path uses a weld/kinematic update. Its separate bimanual demo uses physical friction contacts but only lifts a box. | Model assets need an external checkout; package declares NumPy, h5py, OpenCV, MuJoCo, Torch and TorchVision. Useful controller reference, but physical table-to-table placement needs adaptation. |
| [luckyrobots/g1-manipulation-challenge](https://github.com/luckyrobots/g1-manipulation-challenge) `0e9d1f9b61772905c9f7997b527f221dc7800fc9` | No repository license found; not reusable for code/assets | G1 29 DoF with Dex3, MuJoCo | Exact two-table cylinder scene; walker/croucher/rotator/reacher ONNX assets and manual hand toggle, but no automated pick-place solution | Small MuJoCo/ONNX dependencies and low scene adaptation, but no license and no completed task controller disqualify it as an adoption baseline. |
| [maxwellrobotics/g1-ros2](https://github.com/maxwellrobotics/g1-ros2) `ace298393ec6cadc1f4a66e70a3311e1d2c4d7ff` | BSD-3-Clause | G1 29 DoF with Dex3, `unitree_mujoco` | MoveIt and manipulation actions perform an in-reach cube pick/place; the broader mission also uses locomotion and navigation | ROS 2 Jazzy/Ubuntu 24.04 dev container, MoveIt, GPU-backed simulator and many packages. Exact task is useful evidence but the runtime/ROS integration is out of M0 scope. |
| [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | Apache-2.0 | G1 29 DoF with Dex1 or Dex3, Isaac Sim / Isaac Lab | Existing cylinder pick-place tasks and simulation weights | Exact object/task match, but not MuJoCo; requires Isaac Sim/Isaac Lab and supported NVIDIA RTX hardware, so it is not a compatible baseline. |

Other investigated candidates included Robotics-Ark/ark_unitree_g1 `95775cad397e90ca95198b02c0cbeab4d7418bc3` (MIT; G1/Dex3 MuJoCo model and IK hand-motion sequence, but no object scene or task acceptance) and wiscohumanoids/manipulation_sim (generic arm, not G1; no license found).

REPRO COMMAND: `MUJOCO_GL=egl /tmp/robotsim43-hmg-venv/bin/python scripts/playback_dataset.py /tmp/robotsim43-hmg-drill-demo.hdf5 --action-source wbc-goal --allow-state-divergence --require-task-success --sim-frequency 200 --video-path /tmp/robotsim43-hmg-drill-wbc.mp4` from `/tmp/robotsim43-candidates/humanoidmimicgen`. Runtime: Python 3.10.12, MuJoCo 3.2.6, RoboSuite 1.5.1, NumPy 1.26.4, Torch 2.6.0+cpu, ONNX Runtime 1.22.1, Pinocchio 2.7.0. WBC policy weights are from GR00T-WholeBodyControl commit `4141c34280abb67c82e115342a8720f4a83d750d` and passed upstream SHA-256 checks. Source-demo dataset commit `8553b2736497f41562bdf2476348b815fef68950`; `datasets/06_drill_pnp/demo.hdf5` SHA-256 `bafcb756465c7b7e27a4a091b082fcbd66a355f44a9fbec0987d8ce074e66885`.

WHAT WORKS: Unchanged upstream `LMDrillPnP90_G1_Env` loads the G1 and DrillPnP scene; upstream WBC-goal playback completes 786 actions, reaches the task predicate for 76 steps (first at step 680), finishes with `final_success=true`, and exits 0 in 143.93 seconds. EGL video: `/tmp/robotsim43-hmg-drill-wbc.mp4`; full log: `/tmp/robotsim43-hmg-drill-wbc.log`.

WHAT FAILED: Applying the human-demo's recorded low-level actions did not reproduce the reference physics trajectory (late maximum state divergence was about 2.85) and never reached task success (0/1, final false); this path exited 1 after 141.89 seconds. The upstream README identifies `wbc-goal` as the human-demo behavior path and permits its state divergence. No upstream source was modified.

AUTOMATIC METRICS: RobotSim M0 stage, contact/grasp, lift, transfer, release, settling, finite-state, penetration, and teleport checks are not implemented yet.

FINAL DEMO COMMAND: Pending Phase 3-6.

KNOWN SHORTCUTS: Fixed/planted base and deterministic object/table poses are allowed by Issue #43. The selected baseline's object is a drill, not the M0 bottle/cylinder.

FINAL STATUS: IN PROGRESS.
