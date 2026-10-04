# Issue #43 M0 Status

CURRENT STAGE: Phase 7 complete - reproducible RobotSim M0 bottle pick-and-place demo passes twice with the fixed seed.

CHOSEN BASELINE: ozkannceylan/humanoid_vla bimanual G1 MuJoCo controller.

UPSTREAM REPO/COMMIT: https://github.com/ozkannceylan/humanoid_vla @ 3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12; G1 meshes from https://github.com/unitreerobotics/unitree_mujoco @ 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d.

WHY SELECTED: MIT-licensed source provides the shortest usable path to a G1 29-DoF palm-pad model, physical bimanual contact, Jacobian IK, and PD torque control. RobotSim reuses its PhysicsSim and bimanual trajectory planner, then adds the bottle scene, measured-contact grasp attachment, transfer/lower/release sequence, rendering, and acceptance checks. Unitree meshes retain their separate BSD license.

## Phase 1 Candidate Comparison

| Repository / commit | License | Robot / hand / simulator | Controller, assets, task | Reproduction and adaptation |
|---|---|---|---|---|
| [NVlabs/humanoidmimicgen](https://github.com/NVlabs/humanoidmimicgen) `d82844dcec242c82d6b82628ccc45933c3ad5cbd` | Apache-2.0 source with separately documented RoboCasa/RoboSuite MIT and Unitree BSD assets; downloaded WBC weights have NVIDIA Open Model License | Unitree G1, three-finger hand, MuJoCo 3.2.6 | Nine G1 tasks including Drill PnP; public recorded-action replay and task-success checks; WBC and task data are available separately | Python 3.10; core pins MuJoCo 3.2.6, NumPy 1.26.4, RoboSuite 1.5.1 and robosuite-models 1.0.0. Recorded replay is CPU-capable; public replay data is an external download. Cylinder adaptation is moderate/high. |
| [ozkannceylan/humanoid_vla](https://github.com/ozkannceylan/humanoid_vla) `3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12` | MIT; its G1 meshes come from a separate Unitree checkout | G1 29 DoF with palm pads, MuJoCo 3.x | Scripted Jacobian IK demos cover cube place; ACT checkpoint is not in the checkout and the single-arm path uses a weld/kinematic update. Its separate bimanual demo uses physical friction contacts but only lifts a box. | Model assets need an external checkout; package declares NumPy, h5py, OpenCV, MuJoCo, Torch and TorchVision. Useful controller reference, but physical table-to-table placement needs adaptation. |
| [luckyrobots/g1-manipulation-challenge](https://github.com/luckyrobots/g1-manipulation-challenge) `0e9d1f9b61772905c9f7997b527f221dc7800fc9` | No repository license found; not reusable for code/assets | G1 29 DoF with Dex3, MuJoCo | Exact two-table cylinder scene; walker/croucher/rotator/reacher ONNX assets and manual hand toggle, but no automated pick-place solution | Small MuJoCo/ONNX dependencies and low scene adaptation, but no license and no completed task controller disqualify it as an adoption baseline. |
| [maxwellrobotics/g1-ros2](https://github.com/maxwellrobotics/g1-ros2) `ace298393ec6cadc1f4a66e70a3311e1d2c4d7ff` | BSD-3-Clause | G1 29 DoF with Dex3, `unitree_mujoco` | MoveIt and manipulation actions perform an in-reach cube pick/place; the broader mission also uses locomotion and navigation | ROS 2 Jazzy/Ubuntu 24.04 dev container, MoveIt, GPU-backed simulator and many packages. Exact task is useful evidence but the runtime/ROS integration is out of M0 scope. |
| [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | Apache-2.0 | G1 29 DoF with Dex1 or Dex3, Isaac Sim / Isaac Lab | Existing cylinder pick-place tasks and simulation weights | Exact object/task match, but not MuJoCo; requires Isaac Sim/Isaac Lab and supported NVIDIA RTX hardware, so it is not a compatible baseline. |

Other investigated candidates included Robotics-Ark/ark_unitree_g1 `95775cad397e90ca95198b02c0cbeab4d7418bc3` (MIT; G1/Dex3 MuJoCo model and IK hand-motion sequence, but no object scene or task acceptance) and wiscohumanoids/manipulation_sim (generic arm, not G1; no license found).

REPRO COMMAND: From the repository root in WSL2, run `./scripts/run_m0_pick_place.sh`. For the exact tested external checkouts/output, run `ROBOTSIM_M0_CANDIDATE_DIR=/tmp/robotsim43-candidates/humanoid_vla ROBOTSIM_M0_UNITREE_DIR=/tmp/robotsim-issue12-vendor/unitree_mujoco ROBOTSIM_M0_OUTPUT_DIR=/tmp/robotsim-issue43-m0/output ./scripts/run_m0_pick_place.sh`. The launcher verifies both pinned commits and clean checkout state, creates a Python 3.10 environment under `/tmp/robotsim-issue43-m0`, and writes the generated model, logs, JSON, MP4, and PNG outside the repository.

WHAT WORKS: The unchanged upstream H-VLA bimanual demo lifts its object 7.6 cm over 174 frames. The RobotSim adapter completes GRASP at 2.176 s, LIFT at 4.544 s, TRANSFER at 18.784 s, RELEASE at 24.704 s, and PLACE at 27.328 s (856 control frames; 500 Hz physics, 31.25 Hz control). Two final-code fixed-seed runs produced identical stage times and final state. Final object position is `[0.4528, -0.4876, 1.0998]` m, target XY error 0.2124 m, final linear/angular speeds `4.20e-6 m/s` and `3.34e-6 rad/s`, with target-table contact and a stable side-lying pose. Each run took about 50 s wall time. Video, screenshot, result JSON, and log are in `/tmp/robotsim-issue43-m0/final-verified/` and `/tmp/robotsim-issue43-m0/final-repeat-verified/`.

WHAT FAILED: HMG recorded low-level action replay diverged from its reference trajectory (late maximum state divergence about 2.85) and did not reach task success; its separate WBC-goal replay passed and remains candidate evidence, not the selected adapter. Initial widened-table variants interfered with grasp or prevented stable support and were discarded. No vendor checkout was modified.

AUTOMATIC METRICS: PASS requires bilateral palm force contact (at least 2 N per palm for 5 frames), at least 5 cm lift, object center inside target-table bounds with 3 cm edge clearance, contact-free release for 5 frames, target-table contact, and at least 1.0 s of elapsed stable time (33 consecutive samples, 1.024 s at 31.25 Hz) with linear speed at most 0.03 m/s, angular speed at most 0.20 rad/s, and position spread at most 2 cm. Safety checks cover finite/bounded state, per-frame teleport under 0.20 m, penetration under 2.5 cm, and no drop below 0.5 m. Both final-code fixed-seed runs passed all stages and checks.

FINAL DEMO COMMAND: `./scripts/run_m0_pick_place.sh`. Default artifacts are under `/tmp/robotsim-issue43-m0/output`; use `ROBOTSIM_M0_OUTPUT_DIR` to select another external directory.

KNOWN SHORTCUTS: The G1 base is fixed, object/table poses are deterministic ground truth, and the script uses a measured-contact runtime weld during grasp. The final stable pose is side-lying; upright orientation is reported but is not an acceptance requirement. The adapter runtime pins MuJoCo 3.2.6; the repository's separate G1 smoke suite was validated on MuJoCo 3.3.6.

FINAL STATUS: PASS - ready for final visual review.
